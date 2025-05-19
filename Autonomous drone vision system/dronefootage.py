import cv2

# Load drone video
video_path = r"D:\sathya_nm\footage.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video file not found or couldn't be read.")
    exit()

# Background subtractor to detect moving objects
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Apply Gaussian Blur to reduce noise
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    # Find contours of detected objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 3000 > area > 500:  # Filter objects based on size (ignores tiny & huge objects)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding boxes

    # Show the multi-object tracking output
    cv2.imshow("Drone Detection", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()