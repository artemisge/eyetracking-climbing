import cv2

def convert_timestamp_to_seconds(timestamp):
    # Split the timestamp into minutes and seconds
    minutes, seconds = map(int, timestamp.split(':'))
    # Convert minutes and seconds to total seconds
    total_seconds = minutes * 60 + seconds
    return total_seconds

def cut_video(video_path, start_timestamp, end_timestamp):
    # Convert timestamps to seconds
    start_time = convert_timestamp_to_seconds(start_timestamp)
    end_time = convert_timestamp_to_seconds(end_timestamp)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the frame rate and total number of frames
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate start and end frame indices
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    
    # Set the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path[:-4] + '.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    
    # Read until start frame
    for _ in range(start_frame):
        _, _ = cap.read()
    
    # Read and write frames within the specified range
    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'data/videos/climbing2/world8.mp4'
start_timestamp = '0:53'  # Start time in minutes and seconds
end_timestamp = '1:27'    # End time in minutes and seconds

cut_video(video_path, start_timestamp, end_timestamp)
