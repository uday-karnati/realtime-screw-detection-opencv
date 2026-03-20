# realtime-screw-detection-opencv
Real-time screw presence detection using ArUco markers, homography and Otsu thresholding — OpenCV on Raspberry Pi
# Real-Time Screw Detection — Raspberry Pi + OpenCV

A real-time computer vision system that detects whether screws are 
present or missing in industrial parts using a camera, ArUco markers, 
and OpenCV on a Raspberry Pi.

University project — Ravensburg-Weingarten University of Applied 
Sciences (RWU)

## What it does
- Detects ArUco markers on physical parts to determine position and 
  orientation in real time
- Uses homography to map reference hole positions onto the live 
  camera frame
- Applies Otsu thresholding to classify each hole as SCREW or HOLE
- Draws green circles for detected screws, red for empty holes
- Supports two different part types (ID 0 and ID 1)

## How it works
1. At startup — loads reference images of empty parts and detects 
   hole positions using Hough Circle Transform
2. Camera opens and reads frames continuously
3. Each frame — detects ArUco marker to locate the part
4. Smooths marker position over 5 frames (moving average) to reduce 
   jitter
5. Computes homography matrix to transform reference coordinates into 
   camera coordinates
6. For each hole — extracts a region of interest, applies Gaussian 
   blur + Otsu threshold, counts pixels
7. More white pixels = screw present (bright head reflects light)
8. More black pixels = hole empty (dark cavity)

## Tech stack
- Python 3
- OpenCV (ArUco, HoughCircles, findHomography, Otsu threshold)
- NumPy
- Raspberry Pi camera (1280x720)

## Requirements
pip install opencv-contrib-python numpy

## Run
python screw_detection.py

Press Q to quit.

## Project structure
├── screw_detection.py       # main detection script
├── Part ID0 empty.png       # reference image for part 0
├── Part ID01 empty.png      # reference image for part 1
└── README.md

## Key concepts used
- ArUco marker detection
- Homography and perspective transform
- Hough Circle Transform
- Otsu's thresholding
- Moving average filtering for temporal smoothing
