# Face Detection and Recognition using MediaPipe and FaceNet

This project demonstrates face detection and recognition using MediaPipe for face detection and FaceNet for embedding extraction. The dataset used is the Labeled Faces in the Wild (LFW) dataset.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, you need to clone the repository and install the required dependencies.

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/face-detection-recognition.git
    cd face-detection-recognition
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset

The project uses the Labeled Faces in the Wild (LFW) dataset, which contains images of faces for face recognition tasks. The dataset is automatically fetched using `sklearn.datasets.fetch_lfw_people`.

## Usage

To run the project, follow these steps:

1. Fetch the LFW people dataset:
    ```python
    from sklearn.datasets import fetch_lfw_people
    lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
    ```

2. Extract images and labels:
    ```python
    images = lfw_people.images
    labels = lfw_people.target
    target_names = lfw_people.target_names
    ```

3. Preprocess images:
    ```python
    def preprocess_image(image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        return image_rgb
    ```

4. Detect faces using MediaPipe:
    ```python
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    def detect_faces(image, min_detection_confidence=0.5):
        with mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence) as face_detection:
            image_rgb = preprocess_image(image)
            image_uint8 = image_rgb.astype(np.uint8)
            results = face_detection.process(image_uint8)
            faces = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image_rgb.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    faces.append(bbox)
            return faces
    ```

5. Visualize detected faces:
    ```python
    def visualize_detected_faces(image, faces):
        example_image_bgr = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in faces:
            cv2.rectangle(example_image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plt.imshow(cv2.cvtColor(example_image_bgr, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    ```

6. Initialize FaceNet embedder and extract embeddings:
    ```python
    from keras_facenet import FaceNet
    embedder = FaceNet()

    def extract_embeddings(image, bbox=None):
        if bbox is not None:
            x, y, w, h = bbox
            face = image[y:y+h, x:x+w]
            face_rgb = preprocess_image(face)
        else:
            face_rgb = preprocess_image(image)
        image_resized = cv2.resize(face_rgb, (160, 160))
        mean, std = image_resized.mean(), image_resized.std()
        if std == 0:  # Avoid division by zero
            std = 1
        image_normalized = (image_resized - mean) / std
        embeddings = embedder.embeddings([image_normalized])
        return embeddings[0]
    ```

7. Convert images to embeddings:
    ```python
    import time
    start_time = time.time()
    embeddings = np.array([extract_embeddings(image) for image in images])
    end_time = time.time()
    print(f"Time taken to extract embeddings: {end_time - start_time:.2f} seconds")
    ```

8. Check class distribution and split data into training and testing sets:
    ```python
    from sklearn.model_selection import train_test_split
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42, stratify=labels)
    ```

9. Train and evaluate a k-NN classifier:
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    ```

10. Example face detection and visualization:
    ```python
    example_image = images[1]
    faces = detect_faces(example_image)
    print(f"Example Image: Faces detected: {faces}")
    visualize_detected_faces(example_image, faces)
    ```

11. Extract and match features for detected faces:
    ```python
    for i, bbox in enumerate(faces):
        face_embedding = extract_embeddings(example_image, bbox)
        face_embedding = face_embedding.reshape(1, -1)
        predicted_label = knn.predict(face_embedding)
        predicted_name = target_names[predicted_label[0]]
        print(f"Face {i}: Predicted label: {predicted_name}")
    ```

12. Test face detection on a different image:
    ```python
    different_image_path = 'example.jpg'
    different_image = cv2.imread(different_image_path, cv2.IMREAD_GRAYSCALE)
    if different_image is not None:
        faces = detect_faces(different_image)
        print(f"Non-LFW Image: Faces detected: {faces}")
        visualize_detected_faces(different_image, faces)
    else:
        print("Non-LFW image not found.")
    ```

## Project structure

├── README.md
├── requirements.txt
├── main.py
└── example.jpg
