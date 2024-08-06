import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import av
import json
import os

# Define the emotion dictionary
emotion_dict = {0: 'Angry', 1: 'Calm', 2: 'Happy'}

# Load the pre-trained model
model = load_model("emotion_model_modified.h5")

st.title("Real-time Emotion Recognition")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform emotion recognition
        img, detected_emotion = self.emotion_recog(img, self.model)
        if detected_emotion is not None:
            # Save the last detected emotion to a file
            with open('last_detected_emotion.json', 'w') as f:
                json.dump({"emotion": detected_emotion}, f)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def emotion_recog(self, frame, model, face_cascade_path='haarcascade_frontalface_default.xml'):
        # Prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        detected_emotion = None
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 255), 3)
            
            # Extract the region of interest (ROI) for emotion detection
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            
            # Make prediction on the ROI
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            
            # Annotate the frame with the predicted emotion
            detected_emotion = emotion_dict[maxindex]
            cv2.putText(frame, detected_emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        print(f"Detected emotion: {detected_emotion}")
        return frame, detected_emotion
    
# Function to execute the create playlist script
def playlist():
    with open("playlist.py") as f:
        code = compile(f.read(), "playlist.py", 'exec')
        exec(code, globals())

# Pick a song from the user
def pick_song():
    df = pd.read_csv('filtered_by_mood.csv')

    # Pick 5 random items
    options = df['song_name'].sample(n=5).tolist()

    # Create the dropdown menu
    selected_option = st.selectbox('Select an option:', options)
    
    # Display the selected option
    st.write(f'You selected: {selected_option}')  

    with open('selected_song.json', 'w') as f:
        json.dump({"song": selected_option}, f)


# Function to execute the external script
def clustering_pca():
    with open("song.py") as f:
        code = compile(f.read(), "song.py", 'exec')
        exec(code, globals())
        pick_song() 

webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

if st.button("Proceed"):
    clustering_pca() 
       
if st.button('Make playlist'):
        playlist()      

