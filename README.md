# Real-Time-Object-Detection-using-Deep-Learning-and-Open-CV
This project presents a real-time multi-object detection pipeline using a pre-trained Single Shot Detector (SSD) with MobileNet backbone. Leveraging OpenCV’s Deep Neural Network (DNN) framework, the system performs efficient inference on images and video streams while detecting 80 object categories from the COCO dataset.

This project implements a real-time object detection system using a pre-trained deep learning model integrated with OpenCV’s DNN (Deep Neural Network) module.
The system is capable of detecting and classifying multiple objects simultaneously from images, video files, or live camera streams.

The project demonstrates practical application of computer vision, convolutional neural networks, and deep learning inference pipelines in a real-world scenario.

🔬 Model & Methodology
Model Used

Architecture: Single Shot Detector (SSD)

Backbone Network: MobileNet

Framework: TensorFlow (frozen inference graph)

Inference Engine: OpenCV DNN module

The SSD MobileNet architecture is chosen due to its balance between speed and accuracy, making it suitable for real-time object detection on resource-constrained systems.

Dataset

Dataset: COCO (Common Objects in Context)

Number of Classes: 80

Examples of Classes:
person, car, bicycle, bus, dog, chair, bottle, laptop, cell phone, etc.

The COCO dataset is a standard benchmark dataset widely used in object detection research.

Detection Pipeline

Input image or video frame is captured

Frame is resized and normalized

Forward pass through SSD MobileNet network

Bounding boxes, class IDs, and confidence scores are extracted

Results are visualized with bounding boxes and labels

🧪 Features

✅ Real-time object detection

✅ Supports image, video file, and webcam input

✅ Multi-object detection per frame

✅ Confidence-based filtering

✅ Clean bounding box and label visualization

✅ Lightweight and efficient inference

🛠️ Technologies Used

Python

OpenCV

TensorFlow (pre-trained model)

NumPy

Matplotlib (for visualization in notebooks)

📊 Results

The system successfully detects and classifies multiple objects with high accuracy in real-time.
SSD MobileNet provides efficient inference while maintaining reliable detection performance.

🎯 Applications

Intelligent surveillance systems

Autonomous driving perception modules

Smart traffic monitoring

Human–object interaction analysis

Computer vision research and education

📌 Future Improvements

Use YOLOv8 for higher accuracy

GPU acceleration (CUDA)

Object tracking integration

FPS optimization

Model fine-tuning on custom datasets
