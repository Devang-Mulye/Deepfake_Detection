import cv2
import numpy as np

# Function to perform person detection in a frame
def detect_person(frame):
    # Load pre-trained Haar Cascade classifier for full body detection
    full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform person detection
    persons = full_body_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return persons

# Function to perform skin texture detection within the detected person region
def detect_skin_texture(frame, persons):
    total_frames = 0
    consistent_skin_frames = 0
    
    for (x, y, w, h) in persons:
        # Extract region of interest (ROI) containing the detected person
        person_roi = frame[y:y+h, x:x+w]
        
        # Convert ROI to grayscale
        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        # Compute the Laplacian of the blurred ROI to highlight edges
        laplacian = cv2.Laplacian(blurred_roi, cv2.CV_64F)
        
        # Compute the threshold to detect potential skin texture
        _, skin_mask = cv2.threshold(np.abs(laplacian), 30, 255, cv2.THRESH_BINARY)
        
        # Count the number of white pixels (skin texture) in the mask
        white_pixels = np.sum(skin_mask == 255)
        
        # Threshold to determine if the skin texture is consistent
        if white_pixels < 50000:  # Adjust threshold as needed
            consistent_skin_frames += 1
        
        total_frames += 1
    
    # Calculate the percentage of frames where skin texture is consistent
    consistency_percentage = (consistent_skin_frames / total_frames) * 100
    
    return consistency_percentage

# Main function to process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    consistent_skin_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform person detection in the frame
        persons = detect_person(frame)
        
        # If persons are detected, perform skin texture detection within their regions
        if len(persons) > 0:
            consistency_percentage = detect_skin_texture(frame, persons)
            consistent_skin_frames += consistency_percentage
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    
    # Calculate the percentage of frames where skin texture is consistent across persons
    overall_consistency_percentage = (consistent_skin_frames / total_frames) * 100
    overall_inconsistency_percentage = 100 - overall_consistency_percentage
    
    print(f"Percentage of video frames with consistent skin texture: {overall_consistency_percentage:.2f}%")
    print(f"Percentage of video frames with inconsistent skin texture: {overall_inconsistency_percentage:.2f}%")

# Call the function to process the video
process_video('To_Test/fake4.mp4')