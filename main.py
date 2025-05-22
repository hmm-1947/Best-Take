import cv2
import mediapipe as mp
import os

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

def detect_faces(image_path):
    """Detects faces in an image and returns bounding boxes."""
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not load {image_path}")
        return [], None  # Skip this image if it can't be read
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    faces = []
    if results.detections:
        print(f"✅ Faces detected in {image_path}: {len(results.detections)}")
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            faces.append((x, y, w, h))
    else:
        print(f"⚠️ No faces found in {image_path}")

    return faces, image

# Process all images in a folder
def process_images(folder_path):
    """Detects faces in all images inside a folder and saves them separately."""
    os.makedirs("detected_faces", exist_ok=True)  # Ensure output folder exists

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        faces, image = detect_faces(img_path)

        for i, (x, y, w, h) in enumerate(faces):
            face_img = image[y:y+h, x:x+w]
            save_path = f"detected_faces/{img_name}_face_{i}.jpg"
            cv2.imwrite(save_path, face_img)
            print(f"📸 Saved: {save_path}")

# Run the script on a folder with images
image_folder = "I:\\besttake\\images" 
if os.path.exists(image_folder):
    process_images(image_folder)
else:
    print(f"❌ Error: Folder '{image_folder}' not found!")
