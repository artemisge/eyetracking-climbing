import cv2
heatmap = cv2.imread('heatmap.png')
cv2.imshow('Heatmap', heatmap)
cv2.waitKey(0)