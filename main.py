import cv2
from motpy import Detection, MultiObjectTracker
import torch
import serial
import time
import firebase_admin
from firebase_admin import credentials, db
import time

last_sent_time = time.time()

# Open serial connection between Arduino and Python
try:
    arduino = serial.Serial('COM7', 115200, timeout=1)
    time.sleep(2)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    arduino = None

# Firebase setup
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://mobile-app-starter-ui-default-rtdb.firebaseio.com"
})

def send_to_firebase(voltage, current, power, energy):
    ref = db.reference("monitoring")
    ref.child("voltage").set({"value": voltage})
    ref.child("current").set({"value": current})
    ref.child("power").set({"value": power})
    ref.child("energy").set({"value": energy, "timestamp": time.time()})
    print("Data sent to Firebase individually.")

def send_to_arduino(command):
    if arduino:
        arduino.write(command.encode())
        print(f"Sent to Arduino: {command}")

def firebase_listener(event):
    path = event.path
    data = event.data
    print(f"Path: {path}, Data: {data}")
    
    if path == "/state":
        if data == "ON":
            send_to_arduino("ON")
        elif data == "OFF":
            send_to_arduino("OFF")

control_ref = db.reference("monitoring")
control_ref.listen(firebase_listener)

def draw_boxes(frame, track_results):
    global arduino
    cnt = len(track_results)

    global last_sent_time
    current_time = time.time()

    if arduino and (current_time - last_sent_time >= 5):
        try:
            arduino.write(f"{cnt}\n".encode())
            time.sleep(0.1)
            last_sent_time = current_time

        except serial.SerialException as e:
            print(f"Serial communication error: {e}")

    if arduino:
        response = arduino.readline()
        print(f"Raw Response: {response}")
        try:
            response = response.decode('utf-8').strip()
        except UnicodeDecodeError:
            print("Received non-UTF-8 data, ignoring...")
            response = ""

        if response:
            values = response.split('|')
            if len(values) == 4:
                voltage = float(values[0])
                current = float(values[1])
                power = float(values[2])
                energy = float(values[3])
                print(f"Voltage: {voltage} V, Current: {current} A, Power: {power} W, Energy: {energy} kWh")
                send_to_firebase(voltage, current, power, energy)

    for obj in track_results:
        x, y, w, h = [int(i) for i in obj.box]
        object_id = obj.id
        confidence = obj.score
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {object_id}: {round(confidence, 2)}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.putText(frame, f"People Count: {cnt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

last_sent_time = time.time()

if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    cap = cv2.VideoCapture(0)
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if not available
    tracker = MultiObjectTracker(dt=1 / cap_fps, tracker_kwargs={'max_staleness': 10})
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
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
    if arduino:
        arduino.close()
