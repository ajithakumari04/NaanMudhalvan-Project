import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

marker_id = 84
marker_size = 400  # increased from 200 to 400

# Generate the marker
marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# Add white border around the marker (increased to 100 pixels)
border_size = 100
marker_with_border = cv2.copyMakeBorder(marker_img, border_size, border_size, border_size, border_size,
                                       cv2.BORDER_CONSTANT, value=255)

cv2.imwrite(f"aruco_marker_{marker_id}_border.png", marker_with_border)
print(f"Saved aruco_marker_{marker_id}_border.png")
