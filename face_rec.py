from deepface import DeepFace
import os

def recognize_faces(face_folder):
    """Compares all detected faces and groups similar ones."""
    faces = [os.path.join(face_folder, f) for f in os.listdir(face_folder)]

    grouped_faces = {}  # Dictionary to store grouped faces

    for i, face1 in enumerate(faces):
        if face1 in grouped_faces:
            continue  # Skip if already assigned to a group
        
        grouped_faces[face1] = []  # Create a new group

        for face2 in faces:
            if face1 != face2:
                try:
                    result = DeepFace.verify(face1, face2, model_name="Facenet", enforce_detection=False)
                    if result["verified"]:  # If the same person, add to group
                        grouped_faces[face1].append(face2)
                except:
                    pass  # Ignore errors
        
        print(f"Group for {face1}: {grouped_faces[face1]}")

    return grouped_faces

# Run recognition on the saved faces
groups = recognize_faces("detected_faces")
