Face Recognition Project
***************
Created by Gabriel Baziramwabo
Founder of Benax Technologies Ltd.
***************
1. File System:
The "tree" command typically returns the following file system:
- ├── copy_images_from_clusters.py
- ├── create_clusters.py
- ├── create_dataset.py
- ├── dataset
- │   ├── data.1.1711973742082.jpg
- │   ├── data.1.1711973742672.jpg
- │   ├── data...............
- ├── models
- │   ├── haarcascade_frontalface_default.xml
- │   └── trained_lbph_face_recognizer_model.yml
- ├── predict.py
- ├── predict_and_control.py
- ├── review_cluster.py
- ├── review_dataset.py
- └── train_model.py

2. Environment Configuration:<br>
2.1. Programming Language
   Python3.10 or plus
2.2. Libraries<br>
2.2.1. General AI Project Libraries:
   - os: Operating system interface, for file and directory operations.
   - shutil: High-level file operations, for copying and removing files.
   - numpy: For numerical operations and array manipulation.
   - tensorflow: Deep learning library for building and training neural networks.
   - tensorflow.keras: High-level neural networks API for TensorFlow.
   - sklearn: For machine learning algorithms and utilities.
   - cv2 or opencv-python: OpenCV library for computer vision tasks.
   - time: For time-related operations and delays.
   - serial: For serial communication with devices (e.g., Arduino).

2.2.2. Specific to Face Recognition and Detection:
   - tensorflow.keras.applications: Pre-trained models for image classification tasks (e.g., VGG16).
   - tensorflow.keras.preprocessing.image: For image preprocessing tasks.
   - sklearn.cluster.KMeans: For K-means clustering.
   - tqdm: For progress bars and monitoring loops.
   - PIL (Python Imaging Library): For image processing and manipulation (e.g., converting to grayscale).

<<<<<<< HEAD
3. Procedure:
- 3.1. Create Dataset: "python create_dataset.py". Before recording a face give it an ID.
- 3.2. Review Dataset: "python review_dataset.py". This displays images located in the "dataset" directory one after another.
- 3.3. Create Clusters: "python create_clusters.py". The "dataset" directory is emptied after the clusters are created.
- 3.4. Review Cluster: "python review_cluster.py Clusterk" where k=1,2,3,...n clusters
- 3.5. Manually delete the cluster containing fake faces.
- 3.6. Copy faces from remaining clusters: "python copy_images_from_clusters.py". This returns the images from clusters and then the cluster folders are destroyed. 
- 3.7. Train the model "LBPH Face Recognizer": "python train_model.py". trained_lbph_face_recognizer_model.yml is the file where the trained LBPH Face Recognizer model is serialized and saved. It can be loaded later to perform face recognition on new images without the need to retrain the model every time.
- 3.8. Edit the file "predict.py" to give the person name to the face ID.
- 3.9. Make prediction on the new unseen faces: "python predict.py"
- 3.10. Based on the prediction made, control a device connected to the PC on which this AI progragram is running: "python predict_and_control.py". Look for the file sample_arduino_program.ino" which receives data from the host PC and control an LED accordingly.
=======
---

## 📂 File System

The project’s file structure (simplified):

```
├── 01_create_dataset.py
├── 02_review_dataset.py
├── 03_create_clusters.py
├── 04_review_cluster.py
├── 05_copy_images_from_clusters.py
├── 06_train_model.py
├── 07_predict.py
├── 080_predict_and_control.py
├── 081_sample_arduino_program.ino
├── README.md
├── dataset/
│   ├── data.1.1758632647473.jpg
│   ├── data.1.1758632648135.jpg
│   └── …
└── models/
    ├── haarcascade_frontalface_default.xml
    └── trained_lbph_face_recognizer_model.yml
```

---

## ⚙️ Environment Setup

### 1. Create & Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install opencv-contrib-python==4.10.0.84 numpy Pillow tqdm scikit-learn tensorflow
```

> **Note:**  
> The `cv2.face.LBPHFaceRecognizer_create` function is only available in the **contrib** build of OpenCV (`opencv-contrib-python`).  
> If you accidentally install plain `opencv-python`, uninstall it and reinstall with contrib.

---

## 📚 Libraries Used

### General
- `os` – File and directory operations  
- `shutil` – Copying and removing files  
- `numpy` – Numerical operations & array manipulation  
- `tensorflow` – Deep learning framework  
- `tensorflow.keras` – High-level API for TensorFlow  
- `scikit-learn` – Machine learning utilities  
- `opencv-contrib-python (cv2)` – Computer vision tasks  
- `time` – Time-based operations and delays  
- `serial` – Serial communication with external devices (e.g., Arduino)  

### Face Recognition–Specific
- `tensorflow.keras.applications` – Pre-trained CNN models (e.g., VGG16)  
- `tensorflow.keras.preprocessing.image` – Image preprocessing utilities  
- `sklearn.cluster.KMeans` – Face clustering  
- `tqdm` – Progress bars for loops  
- `Pillow (PIL)` – Image manipulation (e.g., grayscale conversion)  

---

## 🚀 Workflow

### 1. Create Dataset
```bash
python 01_create_dataset.py
```
Assign a unique **ID** to each face before recording.

---

### 2. Review Dataset
```bash
python 02_review_dataset.py
```
Displays images stored in the `dataset/` directory.

---

### 3. Create Clusters
```bash
python 03_create_clusters.py
```
Groups dataset images into clusters.  
> ⚠️ The original `dataset/` folder will be cleared after clustering.

---

### 4. Review Clusters
```bash
python 04_review_cluster.py ClusterK
```
Replace `K` with the cluster number (`1, 2, 3...n`).

---

### 5. Clean Up
Manually delete clusters containing unwanted or invalid faces.

---

### 6. Copy Valid Faces
```bash
python 05_copy_images_from_clusters.py
```
Restores cleaned images from clusters back into `dataset/`.

---

### 7. Train Model (LBPH Face Recognizer)
```bash
python 06_train_model.py
```
- Trains the LBPH model.  
- Saves it as:  
  ```
  models/trained_lbph_face_recognizer_model.yml
  ```
- This file can be reloaded later for predictions without retraining.

---

### 8. Map IDs to Names
Edit:
```python
07_predict.py
```
Assign **names** to the recognized face IDs.

---

### 9. Make Predictions
```bash
python 07_predict.py
```
Runs recognition on unseen faces.

---

### 10. Prediction + Device Control
```bash
python 080_predict_and_control.py
```
- Sends recognition results to external devices (e.g., Arduino).  
- Example Arduino sketch:  
  ```
  081_sample_arduino_program.ino
  ```
  Demonstrates controlling an LED based on recognition results.

---

✅ With this pipeline, you can **capture**, **train**, **predict**, and even **control external hardware** using face recognition.
>>>>>>> d90f095f (introducing env)
