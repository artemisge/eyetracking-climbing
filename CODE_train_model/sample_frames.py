import cv2
import os

def extract_frames(video_path, output_dir, start_time, end_time, sampling_rate):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate start and end frame indices based on timestamps
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames)
    
    # Loop through each frame in the specified interval
    for frame_num in range(start_frame, end_frame, sampling_rate):
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

# # Example usage:
# video_path = "data/test_videos/climbing/trial_1_1/world.mp4"  # Replace with the path to your video file
# output_dir = "data/eye_tracker_framer"  # Directory to save the extracted frames
# start_time = 16  # Start time in seconds
# end_time = 48  # End time in seconds
# sampling_rate = 3  # Sample every 'sampling_rate' frames within the interval

# extract_frames(video_path, output_dir, start_time, end_time, sampling_rate)


def extract_frames_from_videos(input_folder, output_folder, frames_per_second):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through each video file in the input folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp4") or file.endswith(".avi"):  # Ensure video file formats
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]  # Extract video name without extension
                
                # Open the video file
                cap = cv2.VideoCapture(video_path)
                
                # Check if the video file was opened successfully
                if not cap.isOpened():
                    print(f"Error: Unable to open video file {video_path}")
                    continue
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Calculate sampling rate based on desired frames per second
                sampling_rate = int(fps / frames_per_second)
                
                # Loop through each frame in the video
                for frame_num in range(0, total_frames, sampling_rate):
                    # Set the frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    
                    # Read the next frame
                    ret, frame = cap.read()
                    
                    # Check if the frame was read successfully
                    if ret:
                        # Construct the output file path
                        output_path = os.path.join(output_folder, f"{video_name}_frame_{frame_num}.jpg")
                        
                        # Save the frame as an image
                        cv2.imwrite(output_path, frame)
                        print(f"Saved frame {frame_num} from video {video_name}")
                    else:
                        print(f"Error reading frame {frame_num} from video {video_name}")
                
                # Release the video capture object
                cap.release()
                print(f"Frame extraction complete for video {video_name}")

# Example usage:
input_folder = "data/videos/climbing2/world0"  # Folder containing input videos
output_folder = "data/images/homography_test_sequential_frames"      # Output folder to save extracted frames
frames_per_second = 1                         # Frames per second to be extracted

extract_frames_from_videos(input_folder, output_folder, frames_per_second)
