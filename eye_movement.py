import numpy as np
import cv2

# Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('Haarcascade_Face_Detection/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haarcascade_Face_Detection/haarcascade_eye_tree_eyeglasses.xml')

# Variable to store execution state
first_read = True
start_time = 0
blink_count = 0
blink_rate_threshold = 0.2  # Adjust this threshold as needed

# Specify the path to your video file
video_path = "To_Test/fake2.mp4"  # Replace with the actual path

# Create a VideoCapture object for the video file
cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, img = cap.read()

    if not ret:  # If no frame is grabbed, break the loop
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply filter to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Detect faces and process them
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # roi_face is face which is input to eye classifier
            roi_face = gray[y:y + h, x:x + w]
            roi_face_clr = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

            # Examining the length of eyes object for eyes
            if len(eyes) >= 2:
                # Check if program is running for detection
                if first_read:
                    first_read = False
                    start_time = cv2.getTickCount() / cv2.getTickFrequency()
                else:
                    cv2.putText(img,
                                "Eyes open!", (70, 70),
                                cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 255, 255), 2)
            else:
                if not first_read:
                    blink_count += 1
                    print("Blink detected--------------")
    else:
        cv2.putText(img,
                    "No face detected", (100, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 2)

    # Controlling the algorithm with keys
    cv2.imshow('img', img)
    cv2.waitKey(1)

# Calculate the time duration
end_time = cv2.getTickCount() / cv2.getTickFrequency()
duration = end_time - start_time

# Calculate blinking rate
if duration != 0:
    blinking_rate = blink_count / duration
    print("Blinking rate:", blinking_rate)
    if blinking_rate < blink_rate_threshold:
        print("Deepfake detected! Blinking rate is significantly lower than expected.")
    else:
        print("Blinking rate is within normal range.")
else:
    print("No blink detected.")

# Release the VideoCapture object and close any open windows
cap.release()
cv2.destroyAllWindows()
