import cv2
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Load the image
image = cv2.imread("face_image.jpg")
height, width, _ = image.shape

# Convert the image to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform face detection and landmark extraction
results = face_mesh.process(rgb_image)

# Extract the landmarks
landmarks = []
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append((x, y))

# Optional: Draw the landmarks on the image for visualization
for landmark in landmarks:
    print(landmark)
    cv2.circle(image, landmark, 2, (0, 255, 0), -1)

cv2.imshow("Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
