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
        
        # Process video and create output with labels
        start_time = time.time()
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = tempfile.mktemp(suffix='.mp4')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            frame_to_write = frame.copy()
            
            # Process frame at intervals for accident detection
            if frame_count % frame_interval == 0:
                processed_frames += 1
                timestamp = str(timedelta(seconds=int(frame_count / fps)))
                
                status_text.text(f"🔍 Analyzing frame at {timestamp}... ({processed_frames} frames processed)")
                
                # Analyze frame
                is_accident, response = analyze_frame_with_gpt(frame)
                
                if is_accident:
                    # Store accident info
                    accident_frames.append((frame.copy(), timestamp, frame_count))
                    
                    # Add accident label to all frames for next few seconds (show label for 2-3 seconds)
                    label_duration = int(fps * 2)  # Show label for 2 seconds
                    for i in range(max(0, frame_count), min(total_frames, frame_count + label_duration)):
                        if i not in [acc[2] for acc in accident_frames]:  # Avoid duplicate labeling
                            accident_frames.append((None, timestamp, i))
                
                # Small delay to prevent API rate limiting
                time.sleep(0.1)
            
            # Check if current frame should have accident label
            current_accident = None
            for acc_frame, acc_time, acc_frame_num in accident_frames:
                if frame_count >= acc_frame_num and frame_count < acc_frame_num + int(fps * 2):
                    current_accident = acc_time
                    break
            
            # Add label if accident detected
            if current_accident:
                cv2.putText(frame_to_write, "ACCIDENT DETECTED", (30, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame_to_write, f"Time: {current_accident}", (30, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(frame_to_write)
            frame_count += 1
        
        # Release video writer
        out.release()

        # Cleanup
        cap.release()
        
        # Final results
        processing_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Analysis complete! Processed {processed_frames} frames in {processing_time:.1f}s")

        # Display the processed video with accident labels
        if os.path.exists(output_path):
            st.subheader("📹 Processed Video (with accident detection labels)")
            
            # Read the output video file
            with open(output_path, 'rb') as video_file:
                video_bytes = video_file.read()
            
            # Display the video
            st.video(video_bytes)
            
            # Clean up output file
            try:
                os.unlink(output_path)
            except:
                pass

        # Display results summary
        unique_accidents = []
        for acc_frame, acc_time, acc_frame_num in accident_frames:
            if acc_frame is not None:  # Only count actual detections, not label duration frames
                unique_accidents.append((acc_frame, acc_time, acc_frame_num))
        
        if unique_accidents:
            st.success(f"🚨 Found {len(unique_accidents)} accident(s) in the video")
            
            st.subheader("📋 Accident Summary:")
            for i, (frame, timestamp, frame_num) in enumerate(unique_accidents, 1):
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

st.markdown("---")
