import streamlit as st
import openai
import cv2
import tempfile
from PIL import Image
from datetime import timedelta
import base64
import os
from io import BytesIO
import time

# 🔐 Secure API key handling
if 'OPENAI_API_KEY' not in st.secrets:
    st.error("❌ Please add your OpenAI API key to Streamlit secrets")
    st.info("Go to your Streamlit app settings and add OPENAI_API_KEY to secrets")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit UI
st.title("🚗 Crash Detection from Video")
st.write("Upload a video file to analyze for car accidents using AI")

# Fixed configuration
frame_interval = 30  # Analyze every 30 frames (~1 sec if fps ≈ 30)
max_file_size = 25   # Fixed at 25MB

video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Helper to convert frame to base64
def encode_frame_to_base64(frame):
    """Convert OpenCV frame to base64 string for API"""
    try:
        # Resize frame to reduce API costs and improve speed
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        st.error(f"Error encoding frame: {str(e)}")
        return None

def analyze_frame_with_gpt(frame):
    """Analyze frame using GPT Vision API"""
    try:
        base64_img = encode_frame_to_base64(frame)
        if not base64_img:
            return False, "Error encoding frame"
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Analyze this image for car accidents or crashes. Look for: damaged vehicles, vehicles in unusual positions, debris on road, emergency vehicles, or signs of collision. Answer only 'Yes' if you detect an accident, 'No' if normal traffic."
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.1  # Lower temperature for more consistent results
        )
        
        reply = response.choices[0].message.content.strip().lower()
        return "yes" in reply, reply
        
    except Exception as e:
        return False, f"API Error: {str(e)}"

# Process uploaded video
if video_file:
    # Check file size
    file_size_mb = len(video_file.getvalue()) / (1024 * 1024)
    if file_size_mb > max_file_size:
        st.error(f"❌ File too large ({file_size_mb:.1f}MB). Maximum allowed: {max_file_size}MB")
        st.stop()
    
    st.info(f"📁 Processing video ({file_size_mb:.1f}MB)...")
    
    # Create temporary file
    temp_file = None
    try:
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.getvalue())
            temp_file_path = temp_file.name

        # Initialize video capture
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            st.error("❌ Failed to read the video. Please upload a valid video file.")
            st.stop()

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if fps == 0:
            fps = 30  # Fallback
            
        st.info(f"📊 Video Info: {duration:.1f}s, {fps:.1f} FPS, {total_frames} frames")

        # Initialize tracking variables
        accident_frames = []
        frame_count = 0
        processed_frames = 0
        
        # Create UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        video_display = st.empty()
        
        # Process video - store accident detection info
        start_time = time.time()
        accident_detection_map = {}  # frame_number -> timestamp
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # Process frame at intervals for accident detection
            if frame_count % frame_interval == 0:
                processed_frames += 1
                timestamp = str(timedelta(seconds=int(frame_count / fps)))
                
                status_text.text(f"🔍 Analyzing frame at {timestamp}... ({processed_frames} frames processed)")
                
                # Analyze frame
                is_accident, response = analyze_frame_with_gpt(frame)
                
                if is_accident:
                    # Store accident frame info
                    accident_frames.append((frame.copy(), timestamp, frame_count))
                    
                    # Mark frames for labeling (2 seconds duration)
                    label_duration = int(fps * 2)
                    for i in range(frame_count, min(total_frames, frame_count + label_duration)):
                        accident_detection_map[i] = timestamp
                
                # Small delay to prevent API rate limiting
                time.sleep(0.1)

            frame_count += 1

        # Cleanup
        cap.release()
        
        # Final results
        processing_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Analysis complete! Processed {processed_frames} frames in {processing_time:.1f}s")

        # Now create and display the video with accident labels
        if accident_frames:
            st.subheader("📹 Video with Accident Detection")
            
            # Reset video capture to beginning
            cap2 = cv2.VideoCapture(temp_file_path)
            
            # Create output video with FFmpeg-compatible settings
            height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            # Use H.264 codec for better browser compatibility
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            output_path = tempfile.mktemp(suffix='.mp4')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                # Fallback to mp4v if avc1 fails
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            progress_bar2 = st.progress(0)
            status_text2 = st.empty()
            
            while cap2.isOpened():
                ret, frame = cap2.read()
                if not ret:
                    break
                
                progress = frame_count / total_frames
                progress_bar2.progress(progress)
                status_text2.text(f"🎬 Creating output video... Frame {frame_count}/{total_frames}")
                
                # Add accident label if this frame should have one
                if frame_count in accident_detection_map:
                    timestamp = accident_detection_map[frame_count]
                    cv2.putText(frame, "ACCIDENT DETECTED", (30, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(frame, f"Time: {timestamp}", (30, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                out.write(frame)
                frame_count += 1
            
            cap2.release()
            out.release()
            
            # Display the video
            try:
                with open(output_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                
                st.video(video_bytes)
                status_text2.text("✅ Video ready!")
                
            except Exception as e:
                st.error(f"Error displaying video: {str(e)}")
                st.info("Showing original video instead:")
                st.video(video_file.getvalue())
        else:
            st.info("No accidents detected. Showing original video:")
            st.video(video_file.getvalue())

        # Display results summary
        if accident_frames:
            st.success(f"🚨 Found {len(accident_frames)} accident(s) in the video")
            
            st.subheader("📋 Accident Summary:")
            for i, (frame, timestamp, frame_num) in enumerate(accident_frames, 1):
                st.write(f"🔴 **Accident #{i}:** {timestamp} (Frame {frame_num})")
        else:
            st.success("✅ No accidents detected in the video")
            st.balloons()

    except Exception as e:
        st.error("❌ An error occurred while processing the video")
        st.exception(e)
    
    finally:
        # Clean up temporary files
        try:
            if temp_file and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.unlink(output_path)
        except:
            pass  # Ignore cleanup errors

