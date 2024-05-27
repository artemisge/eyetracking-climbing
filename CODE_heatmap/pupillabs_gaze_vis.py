# maybe pupil labs has different frame extracting process and that's why it's not working 

from matplotlib import pyplot as plt
import pandas as pd


csv_file = 'data/Eyetracking_data/001_exported_data/gaze_positions.csv'
gaze = pd.read_csv(csv_file)
# print(gaze.head(5))

FRAME_INDEX = 1601  # Frame index used for visualization

frame_index_path = 'data/Eyetracking_data/extracted_frames_from_recording/'
frame_index_path = frame_index_path + f"frame{str(FRAME_INDEX).rjust(6, '0')}.png"
# print(frame_index_path)

# read frame
frame_index_image = plt.imread(frame_index_path)
# print(frame_index_image.shape)

# Get the array of normalized gaze points for the given index
gaze_points = gaze[gaze["world_index"] == FRAME_INDEX]
gaze_points = gaze_points.sort_values(by="gaze_timestamp")
gaze_points = gaze_points[["norm_pos_x", "norm_pos_y"]]
gaze_points = gaze_points.to_numpy()
print(gaze_points)

# Split gaze points into separate X and Y coordinate arrays
X, Y = gaze_points[:, 0], gaze_points[:, 1]

# Flip the fixation points
# from the original coordinate system,
# where the origin is at botton left,
# to the image coordinate system,
# where the origin is at top left
Y = 1 - Y

# Denormalize gaze points within the frame
H, W = frame_index_image.shape[:-1]
X, Y = X * W, Y * H

print(X, Y)

# Plotting configuration
plt.figure(figsize=(16,9))
plt.title(f"Frame #{FRAME_INDEX}")
plt.axis("off")

# Draw the frame image
plt.imshow(frame_index_image)

# Draw the gaze points for the given frame
plt.scatter(X, Y, color=(0.0, 0.7, 0.25), s=700, alpha=0.2)

# Draw the gaze movement line for the given frame
plt.plot(X, Y, color=(1.0, 0.0, 0.4), lw=3)

plt.show()