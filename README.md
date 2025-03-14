# People Detector
 
People Detector is a Python script that processes videos as input and performs individual people detection, tracking, and counting. It uses [YOLOv5](https://github.com/ultralytics/yolov5), a state-of-the-art deep learning model for object detection, and [motpy](https://github.com/wmuron/motpy), a multi-object tracking library. It then displays bounding boxes around each person, assigns unique IDs, and shows the count of people in the video frame.

## Run People Detector

1. Clone the repository
   
   `git clone https://github.com/scimone/peopledetector.git`
2. Install required packages

   ```
   py -m pip install opencv-python
   py -m pip install motpy
   py -m pip install torch
   py -m pip install pyserial
   py -m pip install git+https://github.com/ultralytics/yolov5.git
   py -m pip install firebase-admin
   ```
3. Place your video file in the `data` folder.  
4. Change into the project directory.
5. Run the main script with `python3 main.py`.
6. Video is displayed with bounding boxes around detected people along with their unique IDs.
"# UCMN-EE-SMART-SWITCHING-AI" 
