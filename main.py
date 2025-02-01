import cv2
from motpy import Detection, MultiObjectTracker
import torch
import serial
import time

# Open serial connection once
arduino = serial.Serial('COM7', 9600, timeout=1)
time.sleep(2)  # Allow time for the connection to establish

def draw_boxes(frame, track_results):
    global arduino
    cnt = len(track_results)

    # Send count as a string followed by a newline
    arduino.write(f"{cnt}\n".encode())

    # Draw bounding boxes
    for obj in track_results:
        x, y, w, h = [int(i) for i in obj.box]
        object_id = obj.id
        confidence = obj.score
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {object_id}: {round(confidence, 2)}", 
                    (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f"People Count: {cnt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    cap = cv2.VideoCapture(0)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = MultiObjectTracker(dt=1 / cap_fps, tracker_kwargs={'max_staleness': 10})

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1020, 500))
        results = model(frame)
        output = results.pandas().xyxy[0]

        objects = output[output['name'] == 'person']
        detections = [Detection(box=[int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])],
                                score=obj['confidence'], class_id=obj['class']) 
                      for _, obj in objects.iterrows()]

        tracker.step(detections=detections)
        track_results = tracker.active_tracks()

        draw_boxes(frame, track_results)
        cv2.imshow('FRAME', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    arduino.close()  # Close serial connection when exiting
