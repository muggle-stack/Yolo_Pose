# Yolo_Pose
## Overview
Yolo_Pose is a project that leverages the ONNX runtime inference engine for real-time pedestrian detection and pose estimation in videos. While Ultralytics provides well-encapsulated APIs, it lacks support for model quantization, which is crucial for optimizing performance on edge devices. This repository aims to fill that gap by offering additional utilities and scripts tailored for quantization and deployment.

## Installation
To get started with Yolo_Pose, follow these steps:

### Clone the repository:
```
git clone https://github.com/muggle-stack/Yolo_Pose.git
```

### Install the required dependencies (make sure you have Python and pip installed):
```
cd Yolo_Pose
pip install -r requirements.txt
```
### Download yolo11 pose models(if you use the ubuntu, please download this model with wget):
```
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
```
If you use windows, click this link to download model, and then move it to the project path.

## Usage
‌Modify the model_path in the code,
And then run this code:
```
python cv_pose.py
```

## Model Quantization
‌If many users want to know about model quantization methods, I will provide the code.‌

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
