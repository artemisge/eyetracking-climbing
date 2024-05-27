import cv2
import os

def extract_frames(video_path, output_dir):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Loop through each frame in the video
    for frame_num in range(total_frames):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Read the next frame
        ret, frame = cap.read()
        
        # Check if the frame was read successfully
        if ret:
            # Construct the output file path
            output_path = os.path.join(output_dir, f"frame_{frame_num}.jpg")
            
            # Save the frame as an image
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {frame_num}")
        else:
            print(f"Error reading frame {frame_num}")
    
    # Release the video capture object
    cap.release()
    print("Frame extraction complete")

# Example usage:
video_path = "data/test_videos/climbing/trial_2_1/phone_v1_1.avi"  # Replace with the path to your video file
output_dir = "data/frames"  # Directory to save the extracted frames
extract_frames(video_path, output_dir)
