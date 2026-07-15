"""
=============================================================
  ARS DRONE PAYLOAD SYSTEM  –  rpi_main.py
=============================================================
  Purpose  : Anomaly Detection Drone Payload
  Sensors  : MPU-6050 (accel + gyro)  |  BMP280 (baro)
  Camera   : Raspberry Pi Camera (Picamera2)
  Detection: YOLOv8 (Ultralytics) – custom anomaly model
  Transport: Flask WiFi stream (MJPEG + JSON API)
  Logging  : CSV flight log + anomaly snapshots saved locally
=============================================================

  QUICK-START CONFIGURATION  ↓  (only section you must edit)
"""

# ─── USER CONFIGURATION ────────────────────────────────────────────────────────

#  WiFi / server
HOST        = "0.0.0.0"   # "0.0.0.0" = all interfaces; or set your Pi's IP
PORT        = 5000

#  YOLO – set path to your anomaly detection model
#    "yolov8n.pt"                    (nano, fastest, generic)
#    "/home/pi/models/anomaly.pt"    (your custom anomaly model)
YOLO_MODEL_PATH   = "yolov8n.pt"       # ← CHANGE THIS to your anomaly model
YOLO_CONF_THRESH  = 0.50               # confidence threshold for anomaly flag
YOLO_ENABLED      = True               # set False to disable detection

#  Anomaly classes your model detects (maps class_id → label)
#  Override with your actual trained class names
ANOMALY_CLASSES = {
    0: "fire",
    1: "smoke",
    2: "flood",
    3: "debris",
    4: "person",
    5: "vehicle",
    6: "structural_damage",
    7: "oil_spill",
}

#  Severity mapping: which class_ids are HIGH severity
HIGH_SEVERITY_CLASSES = {0, 1, 2, 6, 7}   # fire, smoke, flood, damage, oil

#  Camera
CAM_WIDTH    = 640
CAM_HEIGHT   = 480
JPEG_QUALITY = 80

#  Sensor I2C addresses
MPU_ADDR  = 0x68
BMP_ADDR  = 0x76
LPF_ALPHA = 0.85   # low-pass filter strength (0=raw, 1=frozen)

#  Logging
LOG_ENABLED       = True
LOG_DIR           = "/home/pi/flight_logs"
SNAPSHOT_DIR      = "/home/pi/anomaly_snapshots"
LOG_INTERVAL_SEC  = 0.5    # how often to write a CSV row
SNAPSHOT_COOLDOWN = 3.0    # min seconds between snapshots of same class

# ───────────────────────────────────────────────────────────────────────────────

import os
import csv
import math
import time
import threading
import datetime
import numpy as np
import cv2
import smbus
import board
import busio
import adafruit_bmp280
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string, jsonify

# ── Optional YOLO import ──────────────────────────────────────────────────────
yolo_model = None
if YOLO_ENABLED:
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"[YOLO] Model loaded: {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"[YOLO] Failed to load model – detection disabled. Error: {e}")
        YOLO_ENABLED = False

app = Flask(__name__)

# =============================================================================
#  DIRECTORY SETUP
# =============================================================================
if LOG_ENABLED:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

_session_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path     = os.path.join(LOG_DIR, f"flight_{_session_id}.csv") if LOG_ENABLED else None
_snap_dir_ses = os.path.join(SNAPSHOT_DIR, _session_id) if LOG_ENABLED else None
if LOG_ENABLED:
    os.makedirs(_snap_dir_ses, exist_ok=True)

# =============================================================================
#  MPU-6050
# =============================================================================
bus = smbus.SMBus(1)
bus.write_byte_data(MPU_ADDR, 0x6B, 0)   # wake up

def _read_word(addr):
    high = bus.read_byte_data(MPU_ADDR, addr)
    low  = bus.read_byte_data(MPU_ADDR, addr + 1)
    val  = (high << 8) + low
    return val - 65536 if val > 32768 else val

def get_accel():
    return (
        _read_word(0x3B) / 16384.0 * 9.81,
        _read_word(0x3D) / 16384.0 * 9.81,
        _read_word(0x3F) / 16384.0 * 9.81,
    )

def get_gyro():
    return (
        _read_word(0x43) / 131.0,
        _read_word(0x45) / 131.0,
        _read_word(0x47) / 131.0,
    )

# ── Low-pass filter ───────────────────────────────────────────────────────────
_prev_ax = _prev_ay = _prev_az = 0.0
_filter_lock = threading.Lock()

def get_filtered_accel():
    global _prev_ax, _prev_ay, _prev_az
    ax, ay, az = get_accel()
    with _filter_lock:
        ax = LPF_ALPHA * _prev_ax + (1 - LPF_ALPHA) * ax
        ay = LPF_ALPHA * _prev_ay + (1 - LPF_ALPHA) * ay
        az = LPF_ALPHA * _prev_az + (1 - LPF_ALPHA) * az
        _prev_ax, _prev_ay, _prev_az = ax, ay, az
    return ax, ay, az

def get_angles(ax, ay, az):
    roll  = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
    return roll, pitch

# =============================================================================
#  BMP280
# =============================================================================
_i2c   = busio.I2C(board.SCL, board.SDA)
bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(_i2c, address=BMP_ADDR)

print("Calibrating altitude baseline …")
_alts    = [bmp280.altitude for _ in range(20) if not time.sleep(0.1)]
base_alt = sum(_alts) / len(_alts)
print(f"Base altitude: {base_alt:.2f} m")

# =============================================================================
#  SHARED SENSOR STATE
# =============================================================================
sensor_data = {
    "roll": 0.0, "pitch": 0.0,
    "ax": 0.0, "ay": 0.0, "az": 0.0,
    "gx": 0.0, "gy": 0.0, "gz": 0.0,
    "temp": 0.0, "pressure": 0.0, "altitude": 0.0,
}
sensor_lock = threading.Lock()

def sensor_loop():
    while True:
        try:
            ax, ay, az = get_filtered_accel()
            gx, gy, gz = get_gyro()
            roll, pitch = get_angles(ax, ay, az)
            temp     = bmp280.temperature
            pressure = bmp280.pressure
            altitude = bmp280.altitude - base_alt
            with sensor_lock:
                sensor_data.update({
                    "roll":     round(roll, 2),
                    "pitch":    round(pitch, 2),
                    "ax":       round(ax, 2),
                    "ay":       round(ay, 2),
                    "az":       round(az, 2),
                    "gx":       round(gx, 2),
                    "gy":       round(gy, 2),
                    "gz":       round(gz, 2),
                    "temp":     round(temp, 2),
                    "pressure": round(pressure, 2),
                    "altitude": round(altitude, 2),
                })
        except Exception as e:
            print(f"[Sensor] {e}")
        time.sleep(0.05)   # ~20 Hz

threading.Thread(target=sensor_loop, daemon=True).start()

# =============================================================================
#  ANOMALY DETECTION STATE
# =============================================================================
anomaly_state = {
    "active":        False,
    "count":         0,            # total detections this session
    "last_class":    None,
    "last_label":    "—",
    "last_conf":     0.0,
    "last_severity": "NORMAL",     # NORMAL | CAUTION | CRITICAL
    "last_ts":       None,
    "history":       [],           # last 20 anomaly events
    "snapshot_path": None,
}
anomaly_lock  = threading.Lock()
_snap_cooldown_ts = {}            # class_id → last snapshot timestamp

def _severity(cls_id):
    return "CRITICAL" if cls_id in HIGH_SEVERITY_CLASSES else "CAUTION"

def record_anomaly(cls_id, label, conf, snapshot_bgr=None):
    """Thread-safe anomaly recorder; saves snapshot if cooldown passed."""
    global _snap_cooldown_ts
    ts  = datetime.datetime.now()
    sev = _severity(cls_id)

    snap_path = None
    now = time.time()
    if snapshot_bgr is not None and LOG_ENABLED:
        last = _snap_cooldown_ts.get(cls_id, 0)
        if now - last >= SNAPSHOT_COOLDOWN:
            _snap_cooldown_ts[cls_id] = now
            fname = f"{ts.strftime('%H%M%S_%f')}_{label}_{conf:.2f}.jpg"
            snap_path = os.path.join(_snap_dir_ses, fname)
            cv2.imwrite(snap_path, snapshot_bgr)

    evt = {
        "ts":       ts.strftime("%H:%M:%S"),
        "label":    label,
        "conf":     round(conf, 3),
        "severity": sev,
        "snap":     snap_path,
    }

    with anomaly_lock:
        anomaly_state["active"]        = True
        anomaly_state["count"]        += 1
        anomaly_state["last_class"]    = cls_id
        anomaly_state["last_label"]    = label
        anomaly_state["last_conf"]     = round(conf, 3)
        anomaly_state["last_severity"] = sev
        anomaly_state["last_ts"]       = ts.strftime("%H:%M:%S")
        anomaly_state["snapshot_path"] = snap_path
        anomaly_state["history"].append(evt)
        if len(anomaly_state["history"]) > 20:
            anomaly_state["history"].pop(0)

# =============================================================================
#  FLIGHT LOGGER
# =============================================================================
_log_lock     = threading.Lock()
_csv_file     = None
_csv_writer   = None

if LOG_ENABLED:
    _csv_file   = open(_log_path, "w", newline="")
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow([
        "timestamp", "altitude_m", "roll_deg", "pitch_deg",
        "ax", "ay", "az", "gx", "gy", "gz",
        "temp_c", "pressure_hpa",
        "anomaly_active", "anomaly_label", "anomaly_conf", "anomaly_severity",
    ])

def logger_loop():
    while True:
        time.sleep(LOG_INTERVAL_SEC)
        if not LOG_ENABLED:
            continue
        try:
            with sensor_lock:
                sd = dict(sensor_data)
            with anomaly_lock:
                ad = dict(anomaly_state)
            row = [
                datetime.datetime.now().isoformat(),
                sd["altitude"], sd["roll"], sd["pitch"],
                sd["ax"], sd["ay"], sd["az"],
                sd["gx"], sd["gy"], sd["gz"],
                sd["temp"], sd["pressure"],
                ad["active"], ad["last_label"],
                ad["last_conf"], ad["last_severity"],
            ]
            with _log_lock:
                _csv_writer.writerow(row)
                _csv_file.flush()
        except Exception as e:
            print(f"[Logger] {e}")

if LOG_ENABLED:
    threading.Thread(target=logger_loop, daemon=True).start()

# =============================================================================
#  CAMERA + YOLO DETECTION
# =============================================================================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (CAM_WIDTH, CAM_HEIGHT)}
))
picam2.start()
time.sleep(1)

_BOX_COLOURS = {
    "CRITICAL": (0, 34, 255),    # red
    "CAUTION":  (0, 165, 255),   # orange
}
_DEFAULT_COLOUR = (0, 212, 255)  # cyan fallback

def _draw_detections(frame_bgr, results):
    """Draw anomaly detections with severity-coded bounding boxes."""
    detected_anomalies = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])
        label  = ANOMALY_CLASSES.get(cls_id, results[0].names.get(cls_id, str(cls_id)))
        sev    = _severity(cls_id)
        colour = _BOX_COLOURS.get(sev, _DEFAULT_COLOUR)

        # Bounding box
        thickness = 3 if sev == "CRITICAL" else 2
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, thickness)

        # Label background
        txt = f"[{sev}] {label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame_bgr, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        detected_anomalies.append((cls_id, label, conf))

    # Detection count overlay
    count = len(results[0].boxes)
    if count:
        tag = f"ANOMALIES: {count}"
        cv2.putText(frame_bgr, tag, (CAM_WIDTH - 165, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 34, 255), 2, cv2.LINE_AA)

    return detected_anomalies


def generate_frames():
    """MJPEG stream generator with anomaly detection overlay."""
    while True:
        frame     = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if YOLO_ENABLED and yolo_model is not None:
            try:
                results = yolo_model.predict(
                    frame_bgr,
                    conf=YOLO_CONF_THRESH,
                    verbose=False,
                    stream=False,
                )
                detected = _draw_detections(frame_bgr, results)

                # Record highest-confidence anomaly
                if detected:
                    best = max(detected, key=lambda x: x[2])
                    record_anomaly(best[0], best[1], best[2], frame_bgr.copy())
                else:
                    with anomaly_lock:
                        anomaly_state["active"] = False

            except Exception as e:
                print(f"[YOLO Inference] {e}")

        # Timestamp overlay
        ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame_bgr, ts, (8, CAM_HEIGHT - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 212, 255), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode(
            '.jpg', frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ret:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )

# =============================================================================
#  FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/sensors')
def api_sensors():
    with sensor_lock:
        return jsonify(dict(sensor_data))

@app.route('/api/anomaly')
def api_anomaly():
    with anomaly_lock:
        return jsonify({
            "active":    anomaly_state["active"],
            "count":     anomaly_state["count"],
            "label":     anomaly_state["last_label"],
            "conf":      anomaly_state["last_conf"],
            "severity":  anomaly_state["last_severity"],
            "timestamp": anomaly_state["last_ts"],
            "history":   anomaly_state["history"][-10:],  # last 10
        })

@app.route('/api/status')
def api_status():
    return jsonify({
        "yolo_enabled":  YOLO_ENABLED,
        "yolo_model":    YOLO_MODEL_PATH if YOLO_ENABLED else None,
        "cam":           f"{CAM_WIDTH}x{CAM_HEIGHT}",
        "log_enabled":   LOG_ENABLED,
        "session_id":    _session_id,
        "anomaly_count": anomaly_state["count"],
    })

@app.route('/api/full')
def api_full():
    """Combined endpoint — sensors + anomaly in one call."""
    with sensor_lock:
        sd = dict(sensor_data)
    with anomaly_lock:
        ad = {
            "active":    anomaly_state["active"],
            "count":     anomaly_state["count"],
            "label":     anomaly_state["last_label"],
            "conf":      anomaly_state["last_conf"],
            "severity":  anomaly_state["last_severity"],
            "timestamp": anomaly_state["last_ts"],
            "history":   anomaly_state["history"][-10:],
        }
    sd["anomaly"] = ad
    return jsonify(sd)

# =============================================================================
#  MINIMAL HTML (served at / for quick browser check)
# =============================================================================
HTML = r"""
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<title>ARS Drone – Payload System</title>
<style>
  body { background:#050a0f; color:#00d4ff; font-family:monospace; padding:20px; }
  h1   { color:#00ff88; letter-spacing:.2em; }
  a    { color:#00d4ff; }
  img  { border:1px solid #0d3a5c; max-width:640px; display:block; margin-top:10px; }
</style>
</head><body>
<h1>▶ ARS DRONE PAYLOAD</h1>
<p>For the full ground-station GUI, run <code>groundstation_gui.py</code> on your laptop.</p>
<p>API endpoints: &nbsp;
  <a href="/api/full">/api/full</a> &nbsp;|&nbsp;
  <a href="/api/sensors">/api/sensors</a> &nbsp;|&nbsp;
  <a href="/api/anomaly">/api/anomaly</a> &nbsp;|&nbsp;
  <a href="/api/status">/api/status</a>
</p>
<img src="/video_feed" alt="Live Feed">
</body></html>
"""

# =============================================================================
#  RUN
# =============================================================================
if __name__ == "__main__":
    print(f"\n{'='*58}")
    print(f"  ARS DRONE PAYLOAD SYSTEM  –  http://{HOST}:{PORT}")
    print(f"  YOLO   : {'ENABLED  → ' + YOLO_MODEL_PATH if YOLO_ENABLED else 'DISABLED'}")
    print(f"  Cam    : {CAM_WIDTH}×{CAM_HEIGHT}")
    print(f"  Logging: {'ON  → ' + LOG_DIR if LOG_ENABLED else 'OFF'}")
    print(f"  Session: {_session_id}")
    print(f"{'='*58}\n")
    app.run(host=HOST, port=PORT, threaded=True)
