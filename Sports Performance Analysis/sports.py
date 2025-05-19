import cv2
import numpy as np

# Load video file
video_path = r"D:\ak_nm\playvideo.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video file not found or couldn't be read.")
    exit()

# Background subtractor to detect moving objects
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# List to store player positions
player_positions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction (only detects moving objects)
    fg_mask = bg_subtractor.apply(frame)

    # Find contours of detected objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if
         cv2.contourArea(contour) > 2000:  # Ignore small objects
            x, y, w, h = cv2.boundingRect(contour)
            player_positions.append((x + w // 2, y + h // 2))  # Store center position

            # Calculate speed (distance between frames)
            if len(player_positions) > 1:
                prev_x, prev_y = player_positions[-2]
                speed = np.linalg.norm([x - prev_x, y - prev_y])

                # Display speed on-screen
                cv2.putText(frame, f"Speed: {speed:.2f} px/frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 0, 0), 2, cv2.LINE_AA)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show detection output
    cv2.imshow("Player Tracking", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save player positions for heatmap generation
np.save("player_positions.npy", player_positions)