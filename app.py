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

def encode_frame_to_base64(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    input_path = tfile.name

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = 30  # every 1 sec for ~30 fps

    detected_accident_frames = set()
    frame_count = 0
    st.info("⏳ Analyzing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            base64_img = encode_frame_to_base64(frame)

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Does this image show a car accident? Just answer Yes or No."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                            ]
                        }
                    ],
                    max_tokens=10
                )

                reply = response.choices[0].message.content.strip().lower()
                if "yes" in reply:
                    detected_accident_frames.add(frame_count)

            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                break

        frame_count += 1

    cap.release()

    # Write a new video with accident overlays
    st.success("✅ Analysis complete. Rendering output video...")

    cap = cv2.VideoCapture(input_path)
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for acc_frame in detected_accident_frames:
            if frame_count >= acc_frame and frame_count <= acc_frame + frame_interval:
                timestamp = str(timedelta(seconds=int(frame_count / fps)))
                cv2.putText(frame, f"Accident Detected", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                break

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Display the final video
    st.video(output_path)
    st.success("🎞️ Video playback complete.")
