# HelmNet — Safety Helmet Detection (Computer Vision)

CNN-based binary image classifier to detect whether workers in industrial/construction environments are wearing safety helmets. Compares a custom CNN against four VGG16 transfer learning variants on a balanced dataset of 631 images.

## Business Problem
Non-compliance with PPE (personal protective equipment) requirements causes preventable head injuries on construction and industrial sites. Manual monitoring is labour-intensive and inconsistent. An automated vision system can flag violations in real time.

## Dataset
- 631 RGB images, 200×200×3, balanced:
  - `with_helmet`: ~315 images
  - `without_helmet`: ~316 images
- Loaded from NumPy arrays (`images_proj.npy`) with labels in `Labels_proj.csv`

## Approach
1. Data loading and class balance verification — dataset is near-perfectly balanced
2. Sample visualisation across both classes
3. Preprocessing — grayscale conversion, normalisation to [0,1]
4. Data augmentation via `ImageDataGenerator` — rotation, horizontal flips, brightness shifts
5. **Model 1:** Custom CNN — Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense(128) → Dropout → Sigmoid
6. **Models 2–4:** VGG16 frozen base with custom heads (Base, Base+FFNN, Base+FFNN+Dropout, Base+Augmentation)
7. Evaluation — accuracy, recall, precision, F1-score; confusion matrices; misclassified example review

## Results

| Model | Val Accuracy | Val Recall | Val Precision | Val F1 |
|-------|-------------|------------|---------------|--------|
| Custom CNN | 100% | 100% | 100% | 100% |
| VGG16 (Base) | 100% | 100% | 100% | 100% |
| VGG16 (Base+FFNN) | 100% | 100% | 100% | 100% |
| VGG16 (Base+Augmentation) | 100% | 100% | 100% | 100% |
| **Test set** | **100%** | **100%** | **100%** | **100%** |

> Note: Perfect scores reflect the controlled, balanced, high-quality dataset. Real-world deployment requires images with varied lighting, angles, partial obstruction, and camera distances — expect lower performance until the model is fine-tuned on site-specific data.

## Key Findings
- VGG16 transfer learning converged in fewer epochs compared to the custom CNN trained from scratch
- Data augmentation reduced the train/validation loss gap and improved generalisation
- The two classes are visually distinct in this dataset; real-world complexity (hard hats partially visible, different colours, occlusion) would require a harder dataset

## Recommendations
- Collect images from actual deployment sites and fine-tune the model on site-specific data
- Convert to TensorFlow Lite or ONNX for edge deployment on embedded cameras
- Build an alerting pipeline: log non-compliant frames with timestamps and camera IDs for supervisor review

## Technologies
Python · TensorFlow/Keras · NumPy · OpenCV · Matplotlib · Scikit-learn · Jupyter Notebook

## Code
Notebook: [`Project_6_ChinmayRozekar_HelmNet_Full_Code.ipynb`](Project_6_ChinmayRozekar_HelmNet_Full_Code.ipynb)

---
*Author: Chinmay Rozekar*
