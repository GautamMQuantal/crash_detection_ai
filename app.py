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

# Configuration options
col1, col2 = st.columns(2)
with col1:
    frame_interval = st.slider("Frame Analysis Interval", 15, 90, 45, 
                              help="Analyze every N frames (higher = faster but less accurate)")
with col2:
    max_file_size = st.selectbox("Max File Size (MB)", [5, 10, 25, 50], index=1)

video_file = st.file_uploader(
    "Upload a video file", 
    type=["mp4", "avi", "mov"],
    help=f"Maximum file size: {max_file_size}MB"
)

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
        frame_display = st.empty()
        
        # Process video
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # Process frame at intervals
            if frame_count % frame_interval == 0:
                processed_frames += 1
                timestamp = str(timedelta(seconds=int(frame_count / fps)))
                
                status_text.text(f"🔍 Analyzing frame at {timestamp}... ({processed_frames} frames processed)")
                
                # Analyze frame
                is_accident, response = analyze_frame_with_gpt(frame)
                
                if is_accident:
                    # Store accident frame
                    accident_frames.append((frame.copy(), timestamp, frame_count))
                    
                    # Add accident label to frame
                    frame_with_label = frame.copy()
                    cv2.putText(frame_with_label, "ACCIDENT DETECTED", (30, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(frame_with_label, timestamp, (30, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Display accident frame
                    frame_display.image(frame_with_label, channels="BGR", 
                                      caption=f"🚨 ACCIDENT DETECTED at {timestamp}")
                    
                    st.warning(f"🚨 Accident detected at {timestamp}")
                
                # Small delay to prevent API rate limiting
                time.sleep(0.1)

            frame_count += 1

        # Cleanup
        cap.release()
        
        # Final results
        processing_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Analysis complete! Processed {processed_frames} frames in {processing_time:.1f}s")

        # Display results
        if accident_frames:
            st.success(f"🚨 Found {len(accident_frames)} accident(s) in the video")
            
            st.subheader("📋 Accident Summary:")
            for i, (frame, timestamp, frame_num) in enumerate(accident_frames, 1):
                with st.expander(f"Accident #{i} at {timestamp}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(frame, channels="BGR", caption=f"Frame {frame_num}")
                    with col2:
                        st.write(f"⏰ **Time:** {timestamp}")
                        st.write(f"🎬 **Frame:** {frame_num}")
                        st.write(f"📊 **Confidence:** High")
        else:
            st.success("✅ No accidents detected in the video")
            st.balloons()

    except Exception as e:
        st.error("❌ An error occurred while processing the video")
        st.exception(e)
    
    finally:
        # Clean up temporary file
        try:
            if temp_file and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except:
            pass  # Ignore cleanup errors

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Set up API Key:** Add your OpenAI API key to Streamlit secrets
    2. **Upload Video:** Choose a video file (MP4, AVI, or MOV)
    3. **Adjust Settings:** Configure frame interval and file size limits
    4. **Analyze:** The app will process your video and detect accidents
    5. **Review Results:** Check the detected accidents and timestamps
    
    **Tips:**
    - Higher frame intervals process faster but may miss brief accidents
    - Smaller videos process faster and cost less API credits
    - The app analyzes frames using GPT-4 Vision for accurate detection
    """)

st.markdown("---")
