# PPE Detection System 🦺

A real-time Personal Protective Equipment (PPE) detection web application built with Flask, YOLOv5, and WebSockets. Detects safety gear (helmets, vests, etc.) from live camera feeds and video streams.

## Features

- 🎥 Multi-camera live stream monitoring
- 🤖 YOLOv5-based PPE detection model
- 🔐 Login-protected dashboard
- 📡 Real-time video streaming via WebSockets (Flask-SocketIO)
- 🎬 Support for webcam, RTSP streams, and video files

## Project Structure

```
PPE-Detection/
├── app.py                  # Flask app & routes
├── camera_logic.py         # Camera management & YOLOv5 inference
├── find_cameras.py         # Utility to detect connected cameras
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables (copy to .env)
├── templates/
│   ├── index.html          # Main dashboard
│   └── login.html          # Login page
├── models/
│   └── best.pt             # YOLOv5 trained model weights (not included — see below)
└── yolov5/                 # YOLOv5 submodule
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ppe-detection.git
cd ppe-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install python-dotenv      # For environment variable loading
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and fill in your values:
- Generate a secret key: `python -c "import secrets; print(secrets.token_hex(32))"`
- Set a strong admin password

### 5. Add model weights
The trained model file (`models/best.pt`) is not included in this repo due to file size.
- Place it at: `models/best.pt`

### 6. Run the app
```bash
python app.py
```
Visit `http://localhost:5000` and log in with your credentials from `.env`.

## Security Notes

- Never commit your `.env` file — it's listed in `.gitignore`
- Model weights are excluded due to GitHub's 100MB file size limit — host them separately (Google Drive, HuggingFace, or Git LFS)
- Test videos are excluded from the repo for the same reason

## Tech Stack

- **Backend:** Python, Flask, Flask-SocketIO, Flask-Login
- **Detection:** YOLOv5 (Ultralytics), OpenCV, PyTorch
- **Frontend:** HTML/CSS/JS with WebSocket client

## License

MIT License — feel free to use and adapt.
