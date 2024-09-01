import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
from collections import deque

# Load trained-model
model = tf.keras.models.load_model('sign_language_model.keras')

# Dataset Path
dataset_dir = 'asl_dataset/'

# Temporary ImageDataGenerator to get class indices
temp_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
temp_generator = temp_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Get the class indices
class_indices = {v: k for k, v in temp_generator.class_indices.items()}

# Main window
root = tk.Tk()
root.title("Sign Language Detection")

# Global variable to hold the video capture object
cap = None

# Deque to store recent predictions for smoothing
prediction_queue = deque(maxlen=5)

# Function to check if current time is within operational hours
def is_operational():
    current_time = datetime.now().time()
    start_time = datetime.strptime("18:00:00", "%H:%M:%S").time()
    end_time = datetime.strptime("22:00:00", "%H:%M:%S").time()
    return start_time <= current_time <= end_time

# Upload and Process Image Function 
def upload_image():
    if not is_operational():
        result_label.config(text="Application is operational between 6 PM and 10 PM only.")
        return
    
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((64, 64))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        prediction = model.predict(image_array)
        predicted_label_idx = np.argmax(prediction, axis=1)[0]
        print(f"Predicted index: {predicted_label_idx}") 
        
        # Handle case where predicted_label_idx is not in class_indices
        predicted_label = class_indices.get(predicted_label_idx, "Unknown")

        # Display uploaded image and prediction result
        img = ImageTk.PhotoImage(image)
        image_label.config(image=img)
        image_label.image = img
        result_label.config(text=f"Predicted: {predicted_label}")

# Function for real-time video processing
def start_video():
    global cap
    if not is_operational():
        result_label.config(text="Application is operational between 6 PM and 10 PM only.")
        return
    
    cap = cv2.VideoCapture(0)

    def process_frame():
        if cap is None or not cap.isOpened():
            return

        ret, frame = cap.read()
        if not ret:
            return

        # Preprocess the frame
        frame_resized = cv2.resize(frame, (64, 64))
        frame_array = np.array(frame_resized) / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)

        # Predict and smooth predictions
        prediction = model.predict(frame_array)
        predicted_label_idx = np.argmax(prediction, axis=1)[0]
        prediction_queue.append(predicted_label_idx)

        # Use majority voting from the queue for final prediction
        predicted_label_idx = max(set(prediction_queue), key=prediction_queue.count)
        predicted_label = class_indices.get(predicted_label_idx, "Unknown")

        # Display prediction
        result_label.config(text=f"Predicted: {predicted_label}")

        # Convert frame to Tkinter format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_image = ImageTk.PhotoImage(frame_image)
        
        video_label.config(image=frame_image)
        video_label.image = frame_image

        # Skip frames to reduce load
        root.after(30, process_frame)  # Adjust the value to skip more or fewer frames

    process_frame()

# Function to stop video processing
def stop_video():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        video_label.config(image='')
        result_label.config(text="Video stopped.")

# Create UI elements
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

start_video_button = tk.Button(root, text="Start Video", command=start_video)
start_video_button.pack()

stop_video_button = tk.Button(root, text="Stop Video", command=stop_video)
stop_video_button.pack()

image_label = tk.Label(root)
image_label.pack()

video_label = tk.Label(root)
video_label.pack()

result_label = tk.Label(root, text="Result will be displayed here", font=("Helvetica", 16), wraplength=400, pady=20)
result_label.pack()

# Run Application
root.mainloop()
