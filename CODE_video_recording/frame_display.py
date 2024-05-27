import cv2

# Open the video files
et_v = cv2.VideoCapture('data/test_videos/office/world.mp4')
p_v = cv2.VideoCapture('data/test_videos/office/phone_v1.avi')

# Get the frame rates of the videos
et_frame_rate = int(et_v.get(cv2.CAP_PROP_FPS))
p_frame_rate = int(p_v.get(cv2.CAP_PROP_FPS))

# Function to get frame at a specific time
def get_frame_at_time(video_capture, time_seconds):
    # Calculate the frame index corresponding to the given time
    frame_index = int(time_seconds * video_capture.get(cv2.CAP_PROP_FPS))

    # Set the frame index
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame
    ret, frame = video_capture.read()

    return frame

# Ask for time in seconds
time_seconds = float(0)

# Define the time interval between frames (0.5 seconds)
frame_interval = 0.8

# Iterate over frames with the specified time interval
while True:
    # Get frames from each video at the specified time
    et_frame = get_frame_at_time(et_v, time_seconds)
    p_frame = get_frame_at_time(p_v, time_seconds)

    # Resize frames (optional)
    frame_width = 640
    frame_height = 360
    et_frame = cv2.resize(et_frame, (frame_width, frame_height))
    p_frame = cv2.resize(p_frame, (frame_width, frame_height))

    # Display frames
    cv2.imshow('ET Video Frame', et_frame)
    cv2.imshow('Phone Video Frame', p_frame)

    # Wait for the specified time interval (in milliseconds)
    key = cv2.waitKey(int(frame_interval * 1000))

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

    # Increment the time for the next iteration
    time_seconds += frame_interval

# Release video capture objects
et_v.release()
p_v.release()
cv2.destroyAllWindows()
