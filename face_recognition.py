import cv2
import os
import numpy as np
from PIL import Image
from datetime import datetime

# def generate_dataset():
#     """Capture face images and save them in person-specific folders."""
#     # Input validation for person's name
#     while True:
#         person_name = input("Enter the person's name (or 'quit' to exit): ").strip()
#         if not person_name:
#             print("Name cannot be empty!")
#             continue
#         if person_name.lower() == 'quit':
#             return
#         break

def generate_dataset(person_name=None):
    """Capture face images and save them in person-specific folders."""
    # If name not provided, prompt for it
    if person_name is None:
        while True:
            person_name = input("Enter the person's name (or 'quit' to exit): ").strip()
            if not person_name:
                print("Name cannot be empty!")
                continue
            if person_name.lower() == 'quit':
                return 0
            break

    # Create directory structure
    base_dir = "data"
    person_dir = os.path.join(base_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    # Load face detection classifier
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise RuntimeError("Failed to load face detection model")
    except Exception as e:
        print(f"Error loading face detection model: {e}")
        return

    def detect_and_crop_face(img):
        """Detect and return the largest face in the image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) == 0:
            return None
            
        # Get the largest face by area
        (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
        return img[y:y+h, x:x+w]

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return

    sample_count = 0
    max_samples = 200
    img_size = (200, 200)
    min_face_size = 100
    min_face_height = 0.2 * 720  # 20% of frame height

    print(f"\nStarting face collection for {person_name}...")
    print("Press 'ESC' to stop or wait for completion.")

    try:
        while sample_count < max_samples:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame")
                continue

            # Detect, crop and validate face
            cropped_face = detect_and_crop_face(frame)
            
            if cropped_face is not None:
                h, w = cropped_face.shape[:2]
                if h < min_face_height:
                    continue

                # Process face
                gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, img_size)

                # Validate face quality
                if np.mean(resized_face) < 30:
                    print("Face too dark - skipping")
                    continue

                # Save image with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{person_name}_{timestamp}_{sample_count:04d}.jpg"
                filepath = os.path.join(person_dir, filename)
                cv2.imwrite(filepath, resized_face)

                # Display feedback
                display_img = resized_face.copy()
                cv2.putText(display_img, f"{person_name}: {sample_count}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Captured Face", display_img)

                sample_count += 1
                print(f"Saved sample {sample_count}/{max_samples}", end='\r')

            # Show live feed with face detection
            display_frame = frame.copy()
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(min_face_size, min_face_size))
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(display_frame, f"Samples: {sample_count}/{max_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f"Webcam - {person_name} - Press ESC to stop", display_frame)

            if cv2.waitKey(1) == 13:
                break

    # finally:
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     print(f"\nCompleted! Saved {sample_count} face samples in: {person_dir}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nCompleted! Saved {sample_count} face samples in: {person_dir}")
        return sample_count  # Add this line    

def train_classifier(data_dir="data"):
    """
    Train a face recognition classifier using collected face images.
    
    Returns:
        tuple: (recognizer, label_ids) if successful
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    faces = []
    labels = []
    label_ids = {}
    current_id = 0
    min_samples_per_person = 10

    print("\nStarting training process...")
    print("Scanning for training images...")
    
    for root, _, files in os.walk(data_dir):
        if root == data_dir:
            continue
            
        person_name = os.path.basename(root)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"Warning: No images found for {person_name}")
            continue
            
        if person_name not in label_ids:
            label_ids[person_name] = current_id
            current_id += 1
        
        print(f"Found {len(image_files)} images for {person_name}")

        samples_added = 0
        for file in image_files:
            file_path = os.path.join(root, file)
            
            try:
                img = Image.open(file_path).convert('L')
                img_np = np.array(img, 'uint8')
                
                if img_np.mean() < 30:
                    continue
                    
                faces.append(img_np)
                labels.append(label_ids[person_name])
                samples_added += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        if samples_added < min_samples_per_person:
            print(f"Warning: Only {samples_added} usable images for {person_name}")

    if len(faces) == 0:
        raise ValueError("No valid training images found!")
    
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        print("\nWarning: Only one person found in training data.")
    
    print(f"\nTraining on {len(faces)} images from {len(label_ids)} people...")
    
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        
        model_path = "classifier.yml"
        recognizer.save(model_path)
        
        with open("label_mappings.txt", "w") as f:
            for name, id in label_ids.items():
                f.write(f"{id}:{name}\n")
        
        print("\nTraining successful!")
        print(f"Model saved to: {model_path}")
        print("\nPeople recognized:")
        for name in label_ids.keys():
            print(f"- {name}")
        
        return recognizer, label_ids
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

def validate_model(clf, label_map):
    """Check if model is ready for recognition"""
    if not clf or not label_map:
        print("Error: Model not properly loaded!")
        return False
    if len(label_map) < 1:
        print("Warning: Model trained with very few classes!")
    return True

def draw_boundary(img, gray_img, classifier, scaleFactor, minNeighbors, color, clf, label_map, confidence_threshold=77):
    """
    Draw bounding boxes around faces and recognize them.
    """
    faces = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors, minSize=(100, 100))
    coords = []
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        id, confidence = clf.predict(gray_img[y:y+h, x:x+w])
        confidence = int(100 * (1 - confidence / 300))
        
        person_name = label_map.get(id, "UNKNOWN")
        
        if confidence > confidence_threshold:
            cv2.putText(img, f"{person_name} {confidence}%", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(img, "UNKNOWN", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        coords.append([x, y, w, h])
    
    return coords

def recognize(img, clf, face_cascade, label_map, confidence_threshold=77):
    """Recognize faces in the image using the trained classifier."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    draw_boundary(img, gray_img, face_cascade, 1.1, 10, (0, 255, 0), 
                 clf, label_map, confidence_threshold)
    return img

def load_label_map(file_path="label_mappings.txt"):
    """Load label mappings from file."""
    label_map = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                id, name = line.strip().split(':')
                label_map[int(id)] = name
    except FileNotFoundError:
        print(f"Warning: Label mappings file not found at {file_path}")
    return label_map

def run_recognition():
    """Main recognition function with all components initialized."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error loading face detection model!")
        return
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    try:
        clf.read("classifier.yml")
    except:
        print("Error loading classifier model!")
        return
    
    label_map = load_label_map()
    
    if not validate_model(clf, label_map):
        return
    
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not video_capture.isOpened():
        print("Error opening video capture!")
        return
    
    print("Starting face recognition. Press ESC to exit.")
    
    frame_counter = 0
    skip_frames = 2  # Process every 3rd frame
    
    while True:
        ret, img = video_capture.read()
        frame_counter += 1
        
        if frame_counter % skip_frames != 0:
            continue
            
        if not ret:
            print("Error capturing frame")
            break
        
        img = recognize(img, clf, face_cascade, label_map)
        cv2.imshow("Face Recognition", img)
        
        if cv2.waitKey(1) == 13:
            break
    
    video_capture.release()
    cv2.destroyAllWindows()