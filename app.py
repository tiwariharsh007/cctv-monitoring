import cv2
import threading
import time
import random

from flask import Flask, Response, request, jsonify
from flask_socketio import SocketIO

from services.processing import process_frame
from services.alert_service import handle_alert
from reporting import generate_report

# ---------------- INIT ---------------- #
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Camera
cap = cv2.VideoCapture(0)


# ---------------- VIDEO STREAM ---------------- #
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process frame using your pipeline
        frame = process_frame(frame, camera_id="live_cam")

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------- BASIC ROUTES ---------------- #
@app.route('/')
def index():
    return "AI Surveillance System Running 🚀"


# ---------------- REAL-TIME SOCKET DATA ---------------- #
def send_real_time_data():
    while True:
        data = {"value": random.randint(0, 100)}
        socketio.emit('update', data)
        time.sleep(5)


# ---------------- ALERT APIs ---------------- #
@app.route('/api/alert', methods=['POST'])
def receive_alert():
    alert_data = request.json
    print(f"Received alert: {alert_data}")
    return jsonify({'status': 'Alert received'}), 200


@app.route('/api/alerts', methods=['POST'])
def send_alert_api():
    alert_type = request.json.get('alert_type')
    details = request.json.get('details')

    send_advanced_alert(alert_type, details)

    return jsonify({"status": "alert sent"})


# ---------------- REPORT API ---------------- #
@app.route('/api/reports', methods=['GET'])
def get_report():
    data = {
        "columns": ["time", "in", "out"],
        "rows": [["2025-04-25 12:00:00", 5, 2]]
    }

    report_file = generate_report(data, "traffic_report.csv")

    return jsonify({
        "status": "report generated",
        "file": report_file
    })


# ---------------- START SERVER ---------------- #
if __name__ == '__main__':
    # Start background socket thread
    thread = threading.Thread(target=send_real_time_data)
    thread.daemon = True
    thread.start()

    socketio.run(app, host="0.0.0.0", port=5000, debug=True)

    cap.release()