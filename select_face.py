import cv2
import face_recognition
import os
import numpy as np
from skimage import exposure

# Folder containing multiple images
SEARCH_FOLDER = "I:\\Best-Take\\images"
TARGET_IMAGE = "I:\\Best-Take\\images\\20211202_164551.jpg"

# Global variables
selected_face_encoding = None
faces = []
image = None
matched_faces = []
matched_faces_encodings = []
face_positions = []
selected_face_position = None
matched_faces_window_open = False  # Track if window is open

def detect_faces(image_path):
    """Detects faces in an image and returns bounding boxes and encodings."""
    global image, faces
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image_rgb)
    encodings = face_recognition.face_encodings(image_rgb)

    faces.clear()
    for (y, x2, y2, x), encoding in zip(face_locations, encodings):
        faces.append((x, y, x2, y2, encoding))

def draw_faces():
    """Draws bounding boxes on the image."""
    img_copy = image.copy()
    for i, (x, y, x2, y2, _) in enumerate(faces):
        cv2.rectangle(img_copy, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, f"Face {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_copy

def click_event(event, x, y, flags, param):
    """Handles mouse clicks to select a face and search for it."""
    global selected_face_encoding, selected_face_position, matched_faces_window_open
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (fx, fy, fx2, fy2, encoding) in enumerate(faces):
            if fx <= x <= fx2 and fy <= y <= fy2:
                selected_face_encoding = encoding
                selected_face_position = (fx, fy, fx2, fy2)
                print("Searching for similar faces in other images...")
                find_similar_faces(selected_face_encoding)

                if not matched_faces_window_open:
                    show_matched_faces()
                    matched_faces_window_open = True  # Track window state

def find_similar_faces(selected_encoding):
    """Searches for similar faces in the folder and stores them."""
    global matched_faces, matched_faces_encodings, face_positions
    matched_faces.clear()
    matched_faces_encodings.clear()
    face_positions.clear()

    for filename in os.listdir(SEARCH_FOLDER):
        file_path = os.path.join(SEARCH_FOLDER, filename)
        img = cv2.imread(file_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        encodings = face_recognition.face_encodings(img_rgb)

        for (y, x2, y2, x), encoding in zip(face_locations, encodings):
            match = face_recognition.compare_faces([selected_encoding], encoding, tolerance=0.5)
            if match[0]:
                face_crop = img[y:y2, x:x2]  # Crop the face region
                if face_crop.size > 0:
                    matched_faces.append(face_crop)
                    matched_faces_encodings.append(encoding)
                    face_positions.append((x, y, x2, y2))

def show_matched_faces():
    """Displays all matched faces in a separate window and keeps it open."""
    if not matched_faces:
        return

    max_width = 200  # Width of each face thumbnail
    max_height = 200  # Height of each face thumbnail

    resized_faces = [cv2.resize(face, (max_width, max_height)) for face in matched_faces]

    cols = 4
    rows = (len(resized_faces) + cols - 1) // cols

    grid_height = rows * max_height
    grid_width = cols * max_width
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background

    positions = []
    for idx, face in enumerate(resized_faces):
        row = idx // cols
        col = idx % cols
        y_offset = row * max_height
        x_offset = col * max_width
        grid_img[y_offset:y_offset + max_height, x_offset:x_offset + max_width] = face
        positions.append((x_offset, y_offset, x_offset + max_width, y_offset + max_height, idx))

    cv2.imshow("Matched Faces", grid_img)
    cv2.setMouseCallback("Matched Faces", lambda event, x, y, flags, param: swap_face(event, x, y, positions))

def match_histograms(src, ref):
    """Matches color histograms of the source and reference images."""
    matched = exposure.match_histograms(src, ref, channel_axis=-1)  # Use -1 for last axis (RGB)

    return np.clip(matched, 0, 255).astype(np.uint8)

def swap_face(event, x, y, positions):
    global target_image, selected_face_encoding

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Positions Data: {positions}")

        for (px, py, x2, y2, index) in positions:  # Correct unpacking
            cropped_w = x2 - px
            cropped_h = y2 - py
            print(f"px: {px}, py: {py}, width: {cropped_w}, height: {cropped_h}, index: {index}")

            if px <= x <= x2 and py <= y <= y2:  # Correct click detection
                print("Swapping face...")

                # Detect facial landmarks in the target image
                face_landmarks = face_recognition.face_landmarks(target_image)
                if not face_landmarks:
                    print("No facial landmarks detected in the target image.")
                    return
                
                # Create face mask only for key facial features
                mask = np.zeros_like(target_image, dtype=np.uint8)
                for key in ["left_eyebrow", "right_eyebrow", "top_lip", "bottom_lip", "nose_tip"]:
                    points = np.array(face_landmarks[0][key], dtype=np.int32)
                    cv2.fillConvexPoly(mask, points, (255, 255, 255))

                # Resize new face to match original size
                new_face_resized = cv2.resize(cropped_face, (cropped_w, cropped_h))

                # Apply seamless cloning (Poisson blending)
                center = (px + cropped_w // 2, py + cropped_h // 2)  # Center in the target image
                target_image = cv2.seamlessClone(new_face_resized, target_image, mask[:, :, 0], center, cv2.NORMAL_CLONE)

                print("Face swapped successfully!")
                cv2.imshow("Swapped Image", target_image)


# Load image and detect faces
detect_faces(TARGET_IMAGE)

cv2.namedWindow("Select a Face", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select a Face", click_event)

while True:
    img_with_faces = draw_faces()
    cv2.imshow("Select a Face", img_with_faces)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows()
