import cv2
from ultralytics import YOLO
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import webbrowser

# ------------------ Firebase Setup ------------------
cred = credentials.Certificate(r"E:\major\firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

counter_ref = db.collection('metadata').document('counters')
counter_doc = counter_ref.get()
entry_counter = counter_doc.to_dict().get('entry_counter', 1) if counter_doc.exists else 1

# (Optional) Auto open Firestore console in browser
webbrowser.open("https://console.firebase.google.com/project/occupancy-monitoring-28875/firestore/databases/-default-/data/~2Fentries")

# ------------------ YOLOv10 with Tracker ------------------
model = YOLO("yolov10s.pt")   # options: yolov10n.pt, yolov10s.pt, yolov10m.pt

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

people_inside = set()
prev_cx = {}
last_event_time = {}
CROSSING_COOLDOWN_S = 1.2
LINE_POS = 0.40

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    line_x = int(w * LINE_POS)
    cv2.line(frame, (line_x, 0), (line_x, h), (255, 0, 0), 2)

    # ✅ Use YOLO tracking with ByteTrack (smooth IDs)
    results = model.track(
        frame,
        persist=True,
        imgsz=640,
        conf=0.5,
        iou=0.5,
        classes=[0],  # only persons
        tracker="bytetrack.yaml",
        verbose=False
    )

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue
            tid = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2

            # ENTRY
            if prev_cx.get(tid, cx) < line_x <= cx and tid not in people_inside:
                ts = datetime.now().isoformat()
                db.collection('entries').document(str(entry_counter)).set({
                    'track_id': tid, 'event': 'entry', 'timestamp': ts
                })
                entry_counter += 1
                counter_ref.set({'entry_counter': entry_counter})
                people_inside.add(tid)
                print(f"[ENTRY] ID {tid} at {ts}")

            # EXIT
            elif prev_cx.get(tid, cx) > line_x >= cx and tid in people_inside:
                ts = datetime.now().isoformat()
                db.collection('entries').document(str(entry_counter)).set({
                    'track_id': tid, 'event': 'exit', 'timestamp': ts
                })
                entry_counter += 1
                counter_ref.set({'entry_counter': entry_counter})
                people_inside.discard(tid)
                print(f"[EXIT] ID {tid} at {ts}")

            prev_cx[tid] = cx

            # ✅ Draw bounding box + ID (no blinking!)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ✅ Update Firebase live occupancy
    db.collection('occupancy').document('live').set({
        'count': len(people_inside),
        'last_updated': datetime.now().isoformat()
    })

    cv2.putText(frame, f"People Inside: {len(people_inside)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Occupancy Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
