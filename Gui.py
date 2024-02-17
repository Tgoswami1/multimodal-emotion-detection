#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import pyttsx3
import speech_recognition as sr
from PIL import Image, ImageTk

from textblob import TextBlob

# Load the pre-trained model for facial emotion detection
model = model_from_json(open("emotion_model.json", "r").read())
model.load_weights('emotion_model.h5')

# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the text-to-speech engine
text_to_speech_engine = pyttsx3.init()

# Initialize the speech recognition engine
speech_recognition_engine = sr.Recognizer()

# Function to process video frames and detect emotions
def detect_emotions():
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_emotion = emotions[max_index]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, detect_emotions)

# Function to analyze voice tone
def analyze_voice_tone():
    with sr.Microphone() as source:
        text_to_speech_engine.say("Please speak something for voice tone analysis.")
        text_to_speech_engine.runAndWait()
        try:
            print("Listening...")
            audio_data = speech_recognition_engine.listen(source, timeout=5)
            print("Analyzing voice tone...")
            text_to_speech_engine.say("Analyzing voice tone.")
            text_to_speech_engine.runAndWait()
            tone = analyze_voice_tone_helper(audio_data)
            text_to_speech_engine.say(f"Your voice tone is {tone}.")
            text_to_speech_engine.runAndWait()
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")


def analyze_voice_tone_helper(audio_data):
    try:
        text = speech_recognition_engine.recognize_google(audio_data)
        print(f"Recognized Text: {text}")

        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        if sentiment > 0:
            return "Positive"
        elif sentiment < 0:
            return "Negative"
        else:
            return "Neutral"

    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
        return "Unknown"
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return "Error"


def analyze_voice_tone_helper(audio_data):
    # Add your code here to analyze voice tone using the recognized audio data.
    # You may use a third-party library or API for this purpose.
    # Return the analyzed voice tone.
    pass

# Function to close the application
def close_application():
    window.destroy()

# Create the main window
window = tk.Tk()
window.title("Real-Time Emotion Detection")
window.geometry("800x600")

# Create a label to display the video stream
video_label = tk.Label(window)
video_label.pack()

# Create a button to analyze voice tone
voice_tone_button = tk.Button(window, text="Analyze Voice Tone", command=analyze_voice_tone)
voice_tone_button.place(relx=.5, rely=.75)
voice_tone_button.pack()

# Create a button to close the application
close_button = tk.Button(window, text="Close", command=close_application)
close_button.place(relx=.5, rely=.85)
close_button.pack()

# Open the video capture
cap = cv2.VideoCapture(0)

# Start the emotion detection process
detect_emotions()

# Run the Tkinter event loop
window.mainloop()

# Release the video capture and destroy the window
cap.release()
cv2.destroyAllWindows()


# In[ ]:




