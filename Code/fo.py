
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Load the pre-trained face recognition model (replace with your actual model)
model = tf.keras.models.load_model("C:/Users/shaba/Desktop/model_mobile_dropout.h5")

# Define the class labels
class_labels = ["keerthana", "shabari", "shaye shree"]

# Function to update the video feed and make predictions
def update():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        video_label.config(image=photo)
        video_label.photo = photo

        # Preprocess the frame for prediction
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0  # Normalize pixel values
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension

        # Make predictions using the loaded model
        predictions = model.predict(frame)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]

        # Update a label with the predicted class name
        #prediction_label.config(text=f"Predicted Class: {predicted_class}")

    video_label.after(10, update)  # Update every 10 milliseconds

# Function to capture and send the current frame for prediction
def capture_and_predict():
    ret, frame = cap.read()
    if ret:
        # Preprocess the frame for prediction (same as in the update function)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0  # Normalize pixel values
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension

        # Make predictions using the loaded model
        predictions = model.predict(frame)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]

        # Display the predicted class name in the GUI
        prediction_result_label.config(text=f"Captured and Predicted: {predicted_class}")

# Create the main window
root = tk.Tk()
root.title("OpenCV Video Capture with tkinter")

# Create a label for the video feed
video_label = tk.Label(root)
video_label.pack(padx=10, pady=10)

# Create a label to display the predicted class name
#prediction_label = tk.Label(root, text="Predicted Class: Unknown")
#prediction_label.pack(pady=10)

# Create a label to display the prediction result after capturing
prediction_result_label = tk.Label(root, text="Captured and Predicted: Unknown")
prediction_result_label.pack(pady=10)

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

# Create a button to capture the current frame and send it for prediction
capture_button = tk.Button(root, text="Capture and Predict", command=capture_and_predict)
capture_button.pack()

# Start the tkinter main loop
root.after(10, update)  # Start the update function
root.mainloop()

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
