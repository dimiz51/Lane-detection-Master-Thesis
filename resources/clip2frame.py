import cv2
import os

vid_path = '../clips/bad_lanes_croppe.mp4'

# Open the video file
video = cv2.VideoCapture(vid_path)

# Initialize variables
count = 0
time_between_frames = 100 # in milliseconds

# Loop through the video frames
while True:
    # Read a single frame from the video
    ret, frame = video.read()
    
    # If the frame was not read successfully, we've reached the end of the video
    if not ret:
        break
    
    # Resize the frame to 640x480
    resized_frame = cv2.resize(frame, (448, 448))
    
    # Save the frame as an image
    cv2.imwrite(f"../clips/bad_lanes_clip/frame{count}.jpg", resized_frame)
    
    # Increment the frame count
    count += 1
    
    # Set the next frame to read based on the time between frames
    video.set(cv2.CAP_PROP_POS_MSEC, (count * time_between_frames))
