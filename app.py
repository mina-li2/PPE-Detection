from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO
import base64
import cv2
import numpy as np
import os
from dotenv import load_dotenv
from camera_logic import CameraManager

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY is not set! Create a .env file based on .env.example")
 
#previously
#socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet', engineio_logger=True, logger=True)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet', max_http_buffer_size=10_000_000)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')
if not ADMIN_PASSWORD:
    raise RuntimeError("ADMIN_PASSWORD is not set! Create a .env file based on .env.example")
 

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# --- Auth routes ---
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            login_user(User(username))
            return redirect(url_for('index'))
        else:
            error = 'Invalid credentials'
    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    # Stop all camera streams on logout
    for cam_id in list(camera_manager.cameras.keys()):
        camera_manager.stop_stream(cam_id)
    logout_user()
    return redirect(url_for('login'))

# --- Camera manager instance ---
camera_manager = CameraManager(socketio)

@app.route('/api/cameras')
@login_required
def get_cameras():
    return jsonify({'cameras': list(camera_manager.cameras.keys())})

@app.route('/api/add_camera', methods=['POST'])
@login_required
def add_camera():
    data = request.json

    source = data.get('source') or data.get('cam_id')
    if isinstance(source, str):
        source = source.strip().strip('"').strip("'")
        # If it's a plain integer string like "0", "1" — treat as device index
        if source.isdigit():
            source = int(source)
    new_id = camera_manager.add_camera(source)
    return jsonify({'status': 'ok', 'cam_id': new_id})

@app.route('/api/remove_camera', methods=['POST'])
@login_required
def remove_camera():
    data = request.json
    cam_id = data.get('cam_id')
    if isinstance(cam_id, str):
        cam_id = cam_id.strip().strip('"').strip("'")
    try:
        cam_id = int(cam_id)
    except (ValueError, TypeError):
        pass  # keep as string for RTSP/YouTube URLs
    camera_manager.remove_camera(cam_id)
    return jsonify({'status': 'ok'})

# --- SocketIO events ---
@socketio.on('connect')
def on_connect():
    print('Client connected')

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

@socketio.on('start_stream')
def on_start_stream(data):
    cam_id = data.get('cam_id', 0)
    if isinstance(cam_id, str):
        cam_id = cam_id.strip().strip('"').strip("'")
    try:
        cam_id = int(cam_id)
    except (ValueError, TypeError):
        pass  # keep as string for RTSP/YouTube URLs
    print(f'start_stream received for cam {cam_id}')
    camera_manager.start_stream(cam_id)

@socketio.on('stop_stream')
def on_stop_stream(data):
    cam_id = data.get('cam_id', 0)
    if isinstance(cam_id, str):
        cam_id = cam_id.strip().strip('"').strip("'")
    try:
        cam_id = int(cam_id)
    except (ValueError, TypeError):
        pass 
    print(f'stop_stream received for cam {cam_id}')
    camera_manager.stop_stream(cam_id)

if __name__ == '__main__':
    # Auto-add camera 0 on startup
    camera_manager.add_camera(0)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)