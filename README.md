# YOLO-Cat-Segmentation

This repository provides a Python implementation for segmenting images using YOLO11 for instance and semantic segmentation. It specifically detects and segments "cats" from the input image, displaying the results in a combined image with original, instance-segmented, and cat-specific semantic masks.

## Features
- **Original Image:** The original image provided as input.
- **Instance Segmentation:** Detects individual objects and creates masks around them.
- **Cat Semantic Segmentation:** Isolates cats from the rest of the image.

## Output
Here’s the output after running the code:
<img width="1470" height="540" alt="output" src="https://github.com/user-attachments/assets/f1c699fc-ff46-4693-8cf6-73eb6f6d3c65" />

## Requirements
Ensure you have the following dependencies installed:

- `Python 3.x`
- `opencv-python`
- `numpy`
- `ultralytics` (for yolo11x-seg)
- `matplotlib`
