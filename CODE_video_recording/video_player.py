import os
import cv2

# Initialize video capture objects for both videos
et_v = cv2.VideoCapture('data/test_videos/office/world.mp4')
p_v = cv2.VideoCapture('data/test_videos/office/phone_v1.avi')

# Get frame rates of both videos
et_frame_rate = int(et_v.get(cv2.CAP_PROP_FPS))
p_frame_rate = int(p_v.get(cv2.CAP_PROP_FPS))

# Get the dimensions of the frames
frame_width = int(et_v.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(et_v.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'data/test_videos/office/combined_video.avi'
output_video = cv2.VideoWriter(output_path, fourcc, et_frame_rate, (frame_width * 2, frame_height))

while True:
    # Read frames from both videos
    ret1, frame1 = et_v.read()
    ret2, frame2 = p_v.read()

    # Break the loop if frames are not read successfully from both videos
    if not ret1 or not ret2:
        break

    # Resize frames to the same dimensions
    frame1 = cv2.resize(frame1, (frame_width, frame_height))
    frame2 = cv2.resize(frame2, (frame_width, frame_height))

    # Combine frames side by side
    combined_frame = cv2.hconcat([frame1, frame2])

    # Write the combined frame to the output video
    output_video.write(combined_frame)

    # Display the combined frame (optional)
    cv2.imshow('Combined Video', combined_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects and close the output video
et_v.release()
p_v.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
