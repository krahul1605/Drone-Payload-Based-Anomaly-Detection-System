"""
=============================================================
  ARS DRONE  –  GROUND STATION GUI  (groundstation_gui.py)
=============================================================
  Purpose  : Full-featured ground-control display for the
             ARS Drone Payload System (rpi_main.py).
  Features :
    • MJPEG live video feed from RPi (http://<RPI_IP>:5000)
    • YOLO-based anomaly / object detection overlaid locally
    • Real-time sensor dashboard
        – Roll / Pitch gauges
        – Accelerometer  (ax, ay, az)
        – Gyroscope      (gx, gy, gz)
        – Temperature, Pressure, Relative Altitude
    • Detection log (timestamped)
    • Connection status indicator
    • One-click YOLO model selector (yolov8n … yolov8x)
    • Snapshot / recording buttons
    • Artificial horizon widget

  Dependencies:
    pip install opencv-python pillow requests ultralytics numpy tkinter

  Usage:
    python groundstation_gui.py
    ↳ Then type the RPi IP address and click CONNECT.
=============================================================
"""

# ─── USER CONFIGURATION ──────────────────────────────────────────────────────
DEFAULT_RPI_IP   = "192.168.1.100"   # change to your RPi's IP
DEFAULT_PORT     = 5000
POLL_INTERVAL_MS = 100               # sensor poll rate  (ms)
FRAME_INTERVAL   = 30                # ms between frame fetches
YOLO_MODEL       = "yolov8n.pt"      # default model; user can change in GUI
CONF_THRESHOLD   = 0.40              # YOLO confidence threshold
# ─────────────────────────────────────────────────────────────────────────────

import io
import math
import os
import threading
import time
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import requests
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ── optional YOLO ─────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed – YOLO detection disabled.")

# =============================================================================
#  COLOUR PALETTE
# =============================================================================
BG          = "#050a0f"
PANEL       = "#09151e"
ACCENT      = "#00d4ff"
GREEN       = "#00ff88"
ORANGE      = "#ff6b00"
RED         = "#ff2244"
DIM         = "#1a2a36"
TEXT        = "#c8e8f0"
FONT_MONO   = ("Courier New", 10)
FONT_TITLE  = ("Courier New", 11, "bold")
FONT_LARGE  = ("Courier New", 14, "bold")
FONT_TINY   = ("Courier New", 8)

# =============================================================================
#  UTILITIES
# =============================================================================

def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


# =============================================================================
#  ARTIFICIAL HORIZON  (Canvas widget)
# =============================================================================
class ArtificialHorizon(tk.Canvas):
    W, H = 180, 180

    def __init__(self, parent, **kwargs):
        super().__init__(parent, width=self.W, height=self.H,
                         bg=BG, highlightthickness=1,
                         highlightbackground=ACCENT, **kwargs)
        self._roll = 0.0
        self._pitch = 0.0
        self._draw()

    def update_attitude(self, roll, pitch):
        self._roll  = roll
        self._pitch = pitch
        self._draw()

    def _draw(self):
        self.delete("all")
        cx, cy = self.W // 2, self.H // 2
        r = min(cx, cy) - 6

        # clip circle
        self.create_oval(cx - r, cy - r, cx + r, cy + r,
                         fill="#001a2a", outline=ACCENT, width=2)

        # horizon line offset from pitch
        pitch_px = int(self._pitch * (r / 45))
        roll_rad  = math.radians(self._roll)
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)

        # sky / ground halves via rotated polygon
        pts = []
        for angle in range(0, 361, 5):
            a = math.radians(angle)
            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)
            # rotate around center
            dx, dy = x - cx, y - cy
            rx =  dx * cos_r + dy * sin_r + cx
            ry = -dx * sin_r + dy * cos_r + cy + pitch_px
            pts.append((rx, ry))

        # sky above horizon
        h_pts = []
        for i, (x, y) in enumerate(pts):
            if y < cy + pitch_px:
                h_pts.append((x, y))
        if len(h_pts) > 2:
            flat = [c for p in h_pts for c in p]
            self.create_polygon(flat, fill="#003366", outline="")

        # horizon bar
        hx1 = cx + r * math.cos(roll_rad + math.pi / 2)
        hy1 = cy + r * math.sin(roll_rad + math.pi / 2) + pitch_px
        hx2 = cx + r * math.cos(roll_rad - math.pi / 2)
        hy2 = cy + r * math.sin(roll_rad - math.pi / 2) + pitch_px
        self.create_line(hx1, hy1, hx2, hy2, fill=GREEN, width=2)

        # centre reticle
        self.create_line(cx - 20, cy, cx - 6, cy, fill=ACCENT, width=2)
        self.create_line(cx + 6,  cy, cx + 20, cy, fill=ACCENT, width=2)
        self.create_line(cx, cy - 6, cx, cy - 20, fill=ACCENT, width=2)
        self.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                         outline=ACCENT, width=2)

        # labels
        self.create_text(cx, self.H - 8,
                         text=f"R {self._roll:+.1f}°  P {self._pitch:+.1f}°",
                         fill=TEXT, font=FONT_TINY)


# =============================================================================
#  BAR GAUGE  (Canvas widget)
# =============================================================================
class BarGauge(tk.Canvas):
    def __init__(self, parent, label, lo=-20, hi=20,
                 unit="", color=ACCENT, **kwargs):
        super().__init__(parent, width=160, height=22,
                         bg=BG, highlightthickness=0, **kwargs)
        self._lo, self._hi = lo, hi
        self._label = label
        self._unit  = unit
        self._color = color
        self._val   = 0.0
        self._draw()

    def set(self, val):
        self._val = val
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = 160, 22
        ratio = (self._val - self._lo) / (self._hi - self._lo)
        ratio = max(0.0, min(1.0, ratio))
        fill_w = int(ratio * (w - 60))

        self.create_text(2, h // 2, anchor="w",
                         text=f"{self._label}:", fill=TEXT, font=FONT_TINY)
        bar_x = 42
        self.create_rectangle(bar_x, 4, w - 18, h - 4,
                               fill=DIM, outline=DIM)
        bar_color = self._color
        if abs(self._val) > self._hi * 0.85:
            bar_color = RED
        self.create_rectangle(bar_x, 4, bar_x + fill_w, h - 4,
                               fill=bar_color, outline="")
        self.create_text(w - 2, h // 2, anchor="e",
                         text=f"{self._val:+.1f}{self._unit}",
                         fill=TEXT, font=FONT_TINY)


# =============================================================================
#  MAIN GROUND STATION WINDOW
# =============================================================================
class GroundStation(tk.Tk):
    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.title("ARS DRONE  ▶  GROUND STATION")
        self.configure(bg=BG)
        self.resizable(True, True)

        # state
        self._rpi_ip      = tk.StringVar(value=DEFAULT_RPI_IP)
        self._rpi_port    = tk.IntVar(value=DEFAULT_PORT)
        self._connected   = False
        self._yolo_en     = tk.BooleanVar(value=YOLO_AVAILABLE)
        self._model_name  = tk.StringVar(value=YOLO_MODEL)
        self._conf        = tk.DoubleVar(value=CONF_THRESHOLD)
        self._yolo_model  = None
        self._recording   = False
        self._video_writer= None
        self._frame_lock  = threading.Lock()
        self._latest_frame= None          # raw numpy BGR
        self._det_log     = []            # list of strings
        self._session     = requests.Session()
        self._stop_evt    = threading.Event()

        # sensor vars
        self._svars = {k: tk.StringVar(value="—") for k in [
            "roll","pitch","ax","ay","az","gx","gy","gz",
            "temp","pressure","altitude"
        ]}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=BG)
        top.pack(fill=tk.X, padx=6, pady=(6, 0))

        tk.Label(top, text="◈ ARS GROUND STATION", fg=GREEN, bg=BG,
                 font=("Courier New", 16, "bold")).pack(side=tk.LEFT)

        self._status_lbl = tk.Label(top, text="● DISCONNECTED",
                                    fg=RED, bg=BG, font=FONT_TITLE)
        self._status_lbl.pack(side=tk.RIGHT, padx=10)

        # ── connection bar ────────────────────────────────────────────────────
        conn = tk.Frame(self, bg=PANEL, pady=4)
        conn.pack(fill=tk.X, padx=6, pady=4)

        tk.Label(conn, text="RPi IP:", fg=TEXT, bg=PANEL,
                 font=FONT_MONO).pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(conn, textvariable=self._rpi_ip, width=16,
                 bg=DIM, fg=ACCENT, insertbackground=ACCENT,
                 relief=tk.FLAT, font=FONT_MONO).pack(side=tk.LEFT)

        tk.Label(conn, text="Port:", fg=TEXT, bg=PANEL,
                 font=FONT_MONO).pack(side=tk.LEFT, padx=(10, 2))
        tk.Entry(conn, textvariable=self._rpi_port, width=6,
                 bg=DIM, fg=ACCENT, insertbackground=ACCENT,
                 relief=tk.FLAT, font=FONT_MONO).pack(side=tk.LEFT)

        self._conn_btn = tk.Button(
            conn, text="CONNECT", command=self._toggle_connect,
            bg=GREEN, fg=BG, font=FONT_TITLE,
            relief=tk.FLAT, padx=10, cursor="hand2"
        )
        self._conn_btn.pack(side=tk.LEFT, padx=12)

        # YOLO controls
        tk.Label(conn, text="YOLO:", fg=TEXT, bg=PANEL,
                 font=FONT_MONO).pack(side=tk.LEFT, padx=(20, 2))
        yolo_models = ["yolov8n.pt","yolov8s.pt","yolov8m.pt",
                       "yolov8l.pt","yolov8x.pt"]
        ttk.Combobox(conn, textvariable=self._model_name,
                     values=yolo_models, width=12,
                     state="readonly").pack(side=tk.LEFT)

        tk.Checkbutton(conn, text="Enable", variable=self._yolo_en,
                       bg=PANEL, fg=ACCENT, selectcolor=DIM,
                       activebackground=PANEL,
                       command=self._on_yolo_toggle,
                       font=FONT_MONO).pack(side=tk.LEFT, padx=6)

        tk.Label(conn, text="Conf:", fg=TEXT, bg=PANEL,
                 font=FONT_MONO).pack(side=tk.LEFT)
        tk.Scale(conn, from_=0.1, to=0.9, resolution=0.05,
                 orient=tk.HORIZONTAL, variable=self._conf,
                 bg=PANEL, fg=TEXT, troughcolor=DIM,
                 highlightthickness=0, length=100,
                 font=FONT_TINY).pack(side=tk.LEFT)

        # snapshot / record
        tk.Button(conn, text="📷 SNAP", command=self._snapshot,
                  bg=DIM, fg=ACCENT, relief=tk.FLAT,
                  font=FONT_MONO, cursor="hand2").pack(side=tk.RIGHT, padx=4)
        self._rec_btn = tk.Button(conn, text="⏺ REC", command=self._toggle_rec,
                                   bg=DIM, fg=ORANGE, relief=tk.FLAT,
                                   font=FONT_MONO, cursor="hand2")
        self._rec_btn.pack(side=tk.RIGHT, padx=4)

        # ── main body ─────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # left: video
        self._video_frame = tk.Frame(body, bg=BG)
        self._video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._video_lbl = tk.Label(
            self._video_frame,
            text="NO SIGNAL", fg=DIM, bg="#050a0f",
            font=("Courier New", 18, "bold"),
            width=80, height=24,
            relief=tk.FLAT
        )
        self._video_lbl.pack(fill=tk.BOTH, expand=True)

        # detection overlay text on canvas
        self._overlay_lbl = tk.Label(
            self._video_frame,
            text="", fg=GREEN, bg=BG, font=FONT_TINY,
            justify=tk.LEFT
        )
        self._overlay_lbl.pack(anchor=tk.W)

        # right: panels
        right = tk.Frame(body, bg=BG, width=220)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right.pack_propagate(False)

        self._build_horizon_panel(right)
        self._build_imu_panel(right)
        self._build_baro_panel(right)
        self._build_log_panel(right)

        # ── status bar ────────────────────────────────────────────────────────
        sbar = tk.Frame(self, bg=DIM)
        sbar.pack(fill=tk.X, side=tk.BOTTOM)
        self._fps_lbl   = tk.Label(sbar, text="FPS: —", fg=TEXT,
                                    bg=DIM, font=FONT_TINY)
        self._fps_lbl.pack(side=tk.LEFT, padx=8)
        self._det_count = tk.Label(sbar, text="Detections: 0", fg=ACCENT,
                                    bg=DIM, font=FONT_TINY)
        self._det_count.pack(side=tk.LEFT, padx=8)
        tk.Label(sbar,
                 text="ARS Drone Ground Station  ·  YOLO runs locally",
                 fg=DIM, bg=DIM, font=FONT_TINY).pack(side=tk.RIGHT, padx=8)

    # ── panel builders ────────────────────────────────────────────────────────
    def _panel_header(self, parent, title):
        tk.Label(parent, text=title, fg=ACCENT, bg=PANEL,
                 font=FONT_TITLE, anchor="w").pack(
            fill=tk.X, padx=6, pady=(6, 2))
        tk.Frame(parent, bg=ACCENT, height=1).pack(fill=tk.X, padx=6)

    def _build_horizon_panel(self, parent):
        f = tk.Frame(parent, bg=PANEL, relief=tk.FLAT,
                     highlightthickness=1, highlightbackground=DIM)
        f.pack(fill=tk.X, pady=(0, 6))
        self._panel_header(f, "◈ ATTITUDE")
        self._horizon = ArtificialHorizon(f)
        self._horizon.pack(pady=6)

    def _build_imu_panel(self, parent):
        f = tk.Frame(parent, bg=PANEL, relief=tk.FLAT,
                     highlightthickness=1, highlightbackground=DIM)
        f.pack(fill=tk.X, pady=(0, 6))
        self._panel_header(f, "◈ IMU  (MPU-6050)")

        self._gauges = {}
        specs = [
            ("ax", "Ax", -20, 20, " m/s²", ACCENT),
            ("ay", "Ay", -20, 20, " m/s²", ACCENT),
            ("az", "Az",  -2, 20, " m/s²", GREEN),
            ("gx", "Gx", -250,250," °/s",  ORANGE),
            ("gy", "Gy", -250,250," °/s",  ORANGE),
            ("gz", "Gz", -250,250," °/s",  ORANGE),
        ]
        for key, lbl, lo, hi, unit, col in specs:
            g = BarGauge(f, lbl, lo=lo, hi=hi, unit=unit, color=col)
            g.pack(padx=8, pady=1, anchor=tk.W)
            self._gauges[key] = g

    def _build_baro_panel(self, parent):
        f = tk.Frame(parent, bg=PANEL, relief=tk.FLAT,
                     highlightthickness=1, highlightbackground=DIM)
        f.pack(fill=tk.X, pady=(0, 6))
        self._panel_header(f, "◈ BARO  (BMP280)")

        rows = [
            ("temp",     "Temp",     "°C"),
            ("pressure", "Pressure", "hPa"),
            ("altitude", "Rel. Alt", "m"),
        ]
        for key, lbl, unit in rows:
            row = tk.Frame(f, bg=PANEL)
            row.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(row, text=f"{lbl}:", fg=TEXT, bg=PANEL,
                     font=FONT_MONO, width=10, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, textvariable=self._svars[key],
                     fg=GREEN, bg=PANEL, font=FONT_LARGE,
                     anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=unit, fg=DIM, bg=PANEL,
                     font=FONT_TINY).pack(side=tk.LEFT, padx=2)

    def _build_log_panel(self, parent):
        f = tk.Frame(parent, bg=PANEL, relief=tk.FLAT,
                     highlightthickness=1, highlightbackground=DIM)
        f.pack(fill=tk.BOTH, expand=True)
        self._panel_header(f, "◈ DETECTION LOG")

        self._log_text = tk.Text(
            f, bg="#020810", fg=GREEN, font=FONT_TINY,
            relief=tk.FLAT, state=tk.DISABLED,
            wrap=tk.WORD, height=10
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        sb = ttk.Scrollbar(f, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)

        tk.Button(f, text="CLEAR LOG", command=self._clear_log,
                  bg=DIM, fg=TEXT, relief=tk.FLAT,
                  font=FONT_TINY, cursor="hand2").pack(pady=4)

    # ── connection logic ──────────────────────────────────────────────────────
    def _toggle_connect(self):
        if self._connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        ip   = self._rpi_ip.get().strip()
        port = self._rpi_port.get()
        base = f"http://{ip}:{port}"

        # quick reachability test
        try:
            r = self._session.get(f"{base}/api/status", timeout=3)
            r.raise_for_status()
        except Exception as e:
            messagebox.showerror("Connection Failed",
                                 f"Cannot reach {base}\n{e}")
            return

        self._base_url  = base
        self._connected = True
        self._stop_evt.clear()

        self._status_lbl.config(text=f"● CONNECTED  {ip}", fg=GREEN)
        self._conn_btn.config(text="DISCONNECT", bg=RED)

        self._log(f"Connected to {base}")
        self._log(f"YOLO: {'ON' if self._yolo_en.get() else 'OFF'}")

        # load YOLO
        if self._yolo_en.get():
            self._load_yolo()

        # start threads
        threading.Thread(target=self._video_loop,  daemon=True).start()
        threading.Thread(target=self._sensor_loop, daemon=True).start()

    def _disconnect(self):
        self._stop_evt.set()
        self._connected = False
        self._status_lbl.config(text="● DISCONNECTED", fg=RED)
        self._conn_btn.config(text="CONNECT", bg=GREEN)
        self._video_lbl.config(image="", text="NO SIGNAL")
        self._log("Disconnected.")
        if self._recording:
            self._stop_recording()

    # ── YOLO ─────────────────────────────────────────────────────────────────
    def _load_yolo(self):
        if not YOLO_AVAILABLE:
            self._log("[WARN] ultralytics not installed.")
            return
        name = self._model_name.get()
        self._log(f"Loading YOLO model: {name} …")

        def _load():
            try:
                self._yolo_model = _YOLO(name)
                self._log(f"YOLO ready: {name}")
            except Exception as e:
                self._log(f"[ERR] YOLO load failed: {e}")

        threading.Thread(target=_load, daemon=True).start()

    def _on_yolo_toggle(self):
        if self._yolo_en.get() and self._connected:
            self._load_yolo()
        else:
            self._yolo_model = None
            self._log("YOLO disabled.")

    def _run_yolo(self, frame_bgr):
        """Returns (annotated_frame, detection_strings)."""
        if not self._yolo_en.get() or self._yolo_model is None:
            return frame_bgr, []

        conf = self._conf.get()
        results = self._yolo_model(frame_bgr, conf=conf, verbose=False)
        detections = []
        annotated  = frame_bgr.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                score  = float(box.conf[0])
                label  = self._yolo_model.names.get(cls_id, str(cls_id))
                color  = (0, 212, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"{label} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.5, 1)
                cv2.rectangle(annotated,
                              (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              color, -1)
                cv2.putText(annotated, tag, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)
                detections.append(f"[{ts()}] {label} {score:.2f}")

        return annotated, detections

    # ── video loop ────────────────────────────────────────────────────────────
    def _video_loop(self):
        url = f"{self._base_url}/video_feed"
        _t0 = time.time()
        _frames = 0

        try:
            resp = self._session.get(url, stream=True, timeout=10)
        except Exception as e:
            self._log(f"[ERR] video: {e}")
            return

        buf = b""
        for chunk in resp.iter_content(chunk_size=4096):
            if self._stop_evt.is_set():
                break
            buf += chunk

            # extract JPEG from MJPEG
            start = buf.find(b'\xff\xd8')
            end   = buf.find(b'\xff\xd9')
            if start == -1 or end == -1:
                continue

            jpg = buf[start:end + 2]
            buf = buf[end + 2:]

            arr   = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # YOLO
            annotated, dets = self._run_yolo(frame)

            # recording
            if self._recording and self._video_writer is not None:
                self._video_writer.write(annotated)

            # convert for Tk
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail((760, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(pil)

            # schedule GUI update
            det_str = "\n".join(dets[-6:]) if dets else ""
            self.after(0, self._update_video, imgtk, det_str,
                       len(dets))

            # log detections
            for d in dets:
                self._log(d)

            # FPS
            _frames += 1
            elapsed = time.time() - _t0
            if elapsed >= 2.0:
                fps = _frames / elapsed
                self.after(0, self._fps_lbl.config,
                           {"text": f"FPS: {fps:.1f}"})
                _frames = 0
                _t0 = time.time()

    def _update_video(self, imgtk, det_str, n_det):
        self._video_lbl.config(image=imgtk, text="")
        self._video_lbl.imgtk = imgtk          # keep ref
        self._overlay_lbl.config(text=det_str)
        self._det_count.config(text=f"Detections: {n_det}")

    # ── sensor loop ───────────────────────────────────────────────────────────
    def _sensor_loop(self):
        while not self._stop_evt.is_set():
            try:
                r = self._session.get(
                    f"{self._base_url}/api/sensors", timeout=2)
                data = r.json()
                self.after(0, self._update_sensors, data)
            except Exception:
                pass
            time.sleep(POLL_INTERVAL_MS / 1000)

    def _update_sensors(self, data):
        for key, var in self._svars.items():
            val = data.get(key, "—")
            var.set(f"{val}")

        roll  = float(data.get("roll",  0))
        pitch = float(data.get("pitch", 0))
        self._horizon.update_attitude(roll, pitch)

        for key in ("ax","ay","az","gx","gy","gz"):
            val = float(data.get(key, 0))
            if key in self._gauges:
                self._gauges[key].set(val)

    # ── snapshot / recording ──────────────────────────────────────────────────
    def _snapshot(self):
        if not self._connected:
            return
        try:
            r = self._session.get(
                f"{self._base_url}/video_feed",
                stream=True, timeout=5)
            buf = b""
            for chunk in r.iter_content(4096):
                buf += chunk
                s = buf.find(b'\xff\xd8')
                e = buf.find(b'\xff\xd9')
                if s != -1 and e != -1:
                    jpg = buf[s:e+2]
                    break
            else:
                return

            path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG","*.jpg"),("PNG","*.png")],
                initialfile=f"ars_snap_{ts().replace(':','')}.jpg"
            )
            if path:
                with open(path, "wb") as f:
                    f.write(jpg)
                self._log(f"Snapshot saved: {path}")
        except Exception as e:
            self._log(f"[ERR] snapshot: {e}")

    def _toggle_rec(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".avi",
            filetypes=[("AVI","*.avi")],
            initialfile=f"ars_rec_{datetime.datetime.now():%Y%m%d_%H%M%S}.avi"
        )
        if not path:
            return
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._video_writer = cv2.VideoWriter(path, fourcc, 20, (640, 480))
        self._recording = True
        self._rec_btn.config(text="⏹ STOP REC", bg=RED)
        self._log(f"Recording: {path}")

    def _stop_recording(self):
        self._recording = False
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        self._rec_btn.config(text="⏺ REC", bg=DIM)
        self._log("Recording stopped.")

    # ── detection log ─────────────────────────────────────────────────────────
    def _log(self, msg):
        def _do():
            self._log_text.config(state=tk.NORMAL)
            self._log_text.insert(tk.END, msg + "\n")
            self._log_text.see(tk.END)
            self._log_text.config(state=tk.DISABLED)
        self.after(0, _do)

    def _clear_log(self):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)

    # ── close ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._stop_evt.set()
        if self._recording:
            self._stop_recording()
        self.destroy()


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 58)
    print("  ARS DRONE  –  GROUND STATION GUI")
    print(f"  YOLO available : {YOLO_AVAILABLE}")
    print("=" * 58)
    app = GroundStation()
    app.mainloop()