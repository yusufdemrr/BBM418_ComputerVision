# BBM418 - Computer Vision Laboratory Assignments (Spring 2024-2025)

This repository contains solutions for the three programming assignments of the **BBM418 Computer Vision Laboratory** course at Hacettepe University. The projects cover diverse areas in computer vision, including document dewarping, image classification, and object detection.

## Assignment 1: Document Dewarping with Edge and Line Fitting

**Objective:**  
Correct geometric distortions in scanned documents using classical vision techniques.

**Techniques Used:**
- Custom implementation of **Hough Transform** for line detection.
- Robust line fitting with **RANSAC**.
- **Homography**-based geometric transformation for perspective correction.
- **SSIM** (Structural Similarity Index) for evaluation.

**Challenges Addressed:**
- High-resolution image preprocessing.
- Background noise filtering with morphological operations.
- Adaptive ROI masking and contour-based quadrilateral detection.
- Custom post-processing for consistent document boundary detection.

**Result Highlights:**
- Successfully dewarped many documents with mild distortions.
- Struggled with complex curvature and background clutter.
- SSIM used for comparison across 6 distortion types.

See detailed analysis in `Assignment1/report.pdf`.

---

## Assignment 2: Food Image Classification with CNNs and Transfer Learning

**Objective:**  
Classify images of food using both CNNs built from scratch and pre-trained models.

### Part 1: CNNs from Scratch
- Two architectures: Classic CNN and Residual CNN.
- Evaluated multiple learning rates and batch sizes.
- Applied **Dropout** for regularization.
- Visualization: Accuracy/loss plots and confusion matrices.

### Part 2: Transfer Learning
- Used **MobileNetV2** for fine-tuning.
- Compared two strategies:
  - Training only the fully connected layer.
  - Training the last two convolutional blocks + FC layer.

**Result Highlights:**
- Best accuracy from fine-tuning MobileNetV2 (85.82% test accuracy).
- Classic CNN achieved 66.55% accuracy from scratch.
- Transfer learning outperformed all scratch-trained models.

See code and evaluation in `Assignment2/` and full discussion in `Assignment2/report.pdf`.

---

## Assignment 3: Object Detection and Counting in Drone Images with YOLOv8

**Objective:**  
Detect and count cars in aerial parking lot images using **YOLOv8n**.

**Implementation:**
- Trained YOLOv8n under four different freezing strategies:
  - Freeze first 5, 10, 21 blocks, and full training.
- Evaluated with different optimizers and learning rates.
- Used a custom score metric combining precision, recall, and loss.

**Metrics:**
- **Exact Match Accuracy**, **MSE**, **Precision**, and **Recall**.
- Best model (`yolov8n adam lr=0.001 fulltrain`) achieved:
  - **49.00%** Exact Match Accuracy
  - **3.61** Mean Squared Error

**Visualization:**
- Bounding box predictions overlaid on test images.
- Comparative charts and training curves included.

Full results and discussion available in `Assignment3/report.pdf`.

---

## Technologies Used

- Python, NumPy, OpenCV
- PyTorch, torchvision
- Ultralytics YOLOv8
- Jupyter Notebooks
- Kaggle GPU & Google Colab environments

