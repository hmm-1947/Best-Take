import cv2
import face_recognition
import os

# Folder containing multiple images
SEARCH_FOLDER = "I:\\besttake\\images"  # Change this to your folder name
TARGET_IMAGE = "I:\\besttake\\images\\20211202_164551.jpg"  # Image where you select the face

# Global variables
selected_face_encoding = None
faces = []
image = None

def detect_faces(image_path):
    """Detects faces in an image and returns bounding boxes and encodings."""
    global image, faces
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(image_rgb)
    face_locations = face_recognition.face_locations(image_rgb)

    faces = [(x, y, w, h, encoding) for (y, x2, y2, x, encoding) in zip(face_locations, encodings)]

def draw_faces():
    """Draws bounding boxes on the image."""
    img_copy = image.copy()
    for (x, y, w, h, _) in faces:
        cv2.rectangle(img_copy, (x, y), (w, h), (0, 255, 0), 2)
    return img_copy

def find_similar_faces(selected_encoding):
    """Searches for similar faces in the folder."""
    matched_files = []
    
    for filename in os.listdir(SEARCH_FOLDER):
        file_path = os.path.join(SEARCH_FOLDER, filename)
        img = cv2.imread(file_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)

        for encoding in encodings:
            match = face_recognition.compare_faces([selected_encoding], encoding, tolerance=0.5)
            if match[0]:
                matched_files.append(filename)
                break  # Move to the next image once a match is found

    return matched_files

def click_event(event, x, y, flags, param):
    """Handles mouse clicks to select a face and search for it."""
    global selected_face_encoding
    if event == cv2.EVENT_LBUTTONDOWN:
        for (fx, fy, fw, fh, encoding) in faces:
            if fx <= x <= fw and fy <= y <= fh:
                selected_face_encoding = encoding
                print("Searching for similar faces...")
                matches = find_similar_faces(selected_face_encoding)
                print(f"Matching images: {matches}")

# Load image and detect faces
detect_faces(TARGET_IMAGE)

# Show the image with bounding boxes
cv2.namedWindow("Select a Face")
cv2.setMouseCallback("Select a Face", click_event)

while True:
    img_with_faces = draw_faces()
    cv2.imshow("Select a Face", img_with_faces)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows()
