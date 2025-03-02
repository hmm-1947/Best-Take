import cv2
import os
import numpy as np
from deepface import DeepFace

def load_faces(face_folder):
    """Loads all detected faces."""
    return [os.path.join(face_folder, f) for f in os.listdir(face_folder) if f.endswith(('.jpg', '.png'))]

def recognize_faces(faces):
    """Compares faces and groups them by identity."""
    grouped_faces = {}
    
    for i, face1 in enumerate(faces):
        if face1 in grouped_faces:
            continue  # Skip if already assigned
        
        grouped_faces[face1] = []  # Create a new group
        
        for face2 in faces:
            if face1 != face2:
                try:
                    result = DeepFace.verify(face1, face2, model_name="Facenet", enforce_detection=False)
                    if result["verified"]:
                        grouped_faces[face1].append(face2)
                except:
                    pass

    return grouped_faces

def show_faces_grid(grouped_faces):
    """Displays the grouped faces in a grid layout."""
    for main_face, similar_faces in grouped_faces.items():
        images = [main_face] + similar_faces  # Include main face in the group
        face_imgs = [cv2.imread(img) for img in images if os.path.exists(img)]
        
        if not face_imgs:
            continue

        # Resize faces for display
        face_imgs = [cv2.resize(img, (100, 100)) for img in face_imgs]

        # Create grid layout
        grid_width = min(len(face_imgs), 5)
        grid_height = (len(face_imgs) + grid_width - 1) // grid_width
        blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Arrange images into rows
        rows = []
        for i in range(0, len(face_imgs), grid_width):
            row_imgs = face_imgs[i:i+grid_width]
            while len(row_imgs) < grid_width:
                row_imgs.append(blank_img)  # Fill with blanks if needed
            rows.append(np.hstack(row_imgs))

        # Stack rows together
        final_grid = np.vstack(rows)
        cv2.imshow(f"Group {main_face}", final_grid)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the visualization
face_folder = "detected_faces"
faces = load_faces(face_folder)
grouped_faces = recognize_faces(faces)
show_faces_grid(grouped_faces)
