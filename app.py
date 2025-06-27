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

openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("🚗 Crash Detection from Video")
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Helper: frame to base64
def encode_frame_to_base64(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if video_file:
    with st.spinner("⏳ Analyzing video..."):
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 30  # Check 1 per second if ~30 fps

        detected_accident_frames = set()
        frame_count = 0

        # Phase 1: Analyze every nth frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                base64_img = encode_frame_to_base64(frame)

                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[{
                            "role": "user",
                            "content": "Does this image show a car accident? Just answer Yes or No."
                        }],
                        max_tokens=10
                    )

                    # Corrected way to access the response
                    reply = response['choices'][0]['message']['content'].strip().lower()
                    if "yes" in reply:
                        detected_accident_frames.add(frame_count)

                except Exception as e:
                    st.error(f"OpenAI API error: {e}")
                    break

            frame_count += 1

        cap.release()

        # Phase 2: Replay full video with overlay detection and save to an output file
        st.success("✅ Analysis complete. Replaying video with labels...")

        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 file
        out = cv2.VideoWriter(output_file.name, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # If accident detected earlier near this frame, overlay label
            for acc_frame in detected_accident_frames:
                if frame_count >= acc_frame and frame_count <= acc_frame + frame_interval:
                    timestamp = str(timedelta(seconds=int(frame_count / fps)))
                    cv2.putText(frame, f"Accident Detected", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    break

            out.write(frame)  # Write frame to output video file
            frame_count += 1

        cap.release()
        out.release()

        # Display the output video using Streamlit
        st.success("🎞️ Video playback complete.")
        st.video(output_file.name)
