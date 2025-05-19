import cv2
import cv2.aruco as aruco
import numpy as np

# === Load the ArUco marker image ===
image_path = "aruco_marker_84_border.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# === Set dictionary and parameters ===
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# === Detect markers ===
corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

if ids is not None:
    print("Detected marker IDs:", ids.flatten())
    aruco.drawDetectedMarkers(img_color, corners, ids)

    # Image dimensions
    h, w = img.shape

    # === Camera matrix ===
    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # No distortion

    marker_length = 0.2  # Marker size in some real-world unit (meters)
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    for i in range(len(ids)):
        rvec = rvecs[i]
        tvec = tvecs[i]

        # === Draw axis ===
        cv2.drawFrameAxes(img_color, camera_matrix, dist_coeffs, rvec, tvec, 0.1)


        # === Define centered cube coordinates ===
        s = marker_length / 2
        cube_points = np.float32([
            [-s, -s, 0],
            [-s, s, 0],
            [s, s, 0],
            [s, -s, 0],
            [-s, -s, -marker_length],
            [-s, s, -marker_length],
            [s, s, -marker_length],
            [s, -s, -marker_length]
        ])
        imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # Draw base
        cv2.drawContours(img_color, [imgpts[:4]], -1, (0, 255, 255), 2)
        # Draw top
        cv2.drawContours(img_color, [imgpts[4:]], -1, (0, 255, 0), 2)
        # Draw vertical lines
        for j in range(4):
            cv2.line(img_color, tuple(imgpts[j]), tuple(imgpts[j + 4]), (255, 255, 0), 2)

        # Marker ID
        c = corners[i][0]
        top_left = tuple(np.int32(c[0]))
        for i, c in enumerate(corners):
          id_val = ids[i][0]
          top_left = c[0][0]
          text_pos = (int(top_left[0]), int(top_left[1] - 10))  # shift up 10 pixels

        cv2.putText(img_color, f"id: {id_val}", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

else:
    print("No markers detected.")

cv2.imshow("3D Cube Centered", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
