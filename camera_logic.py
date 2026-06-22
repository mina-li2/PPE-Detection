import cv2
import torch
import numpy as np
import base64
import threading
import time
import sys
import os
import yt_dlp

def resolve_stream_url(source):
    if isinstance(source, str) and ('youtube.com' in source or 'youtu.be' in source):
        print(f'Resolving YouTube stream: {source}')
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                return info['url']
        except Exception as e:
            print(f'Failed to resolve YouTube URL: {e}')
            return source
    return source


YOLOV5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
sys.path.insert(0, YOLOV5_PATH)

VIOLATION_CLASSES = ['no-boots', 'no-gloves', 'no-goggles', 'no-helmet', 'no-vest']


class CameraStream:
    def __init__(self, source, cam_id, model, socketio):
        self.source = source
        self.cam_id = cam_id
        self.model = model
        self.socketio = socketio
        self.cap = None
        self.running = False
        self.alert_cooldown = {}

    def open_capture(self):
        stream_url = resolve_stream_url(self.source)
        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            print(f'Cannot open camera {self.cam_id}')
            return False
        self.running = True
        print(f'Camera {self.cam_id} capture opened successfully')
        return True

    def stop(self):
        self.running = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
            self.cap = None
        print(f'Camera {self.cam_id} stopped')

    def _loop(self):
        print(f'Stream loop started for camera {self.cam_id}')

        if not self.cap or not self.cap.isOpened():
            print(f'Camera {self.cam_id}: capture not open, aborting loop')
            return
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print(f'Camera {self.cam_id}: failed to read frame')
                    break
                

                frame = cv2.resize(frame, (640, 480))
                results = self.model(frame)
                detections = results.pandas().xyxy[0]

                ppe_classes = ['boots', 'gloves', 'goggles', 'helmet', 'vest']
                violation_classes = ['no-boots', 'no-gloves', 'no-goggles', 'no-helmet', 'no-vest']

                
                if 'person' in detections['name'].values:
                    person_count = len(detections[detections['name'] == 'person'])
                else:
                    all_det = detections[detections['name'].isin(ppe_classes + violation_classes)]
                    if len(all_det) == 0:
                        person_count = 0
                    else:
                        centers = []
                        for _, row in all_det.iterrows():
                            cx = (row.xmin + row.xmax) / 2
                            cy = (row.ymin + row.ymax) / 2
                            centers.append((cx, cy))

                        clusters = []
                        for cx, cy in centers:
                            found = False
                            for cluster in clusters:
                                for bx, by in cluster:
                                    if abs(cx - bx) < 150 and abs(cy - by) < 150:
                                        cluster.append((cx, cy))
                                        found = True
                                        break
                                if found:
                                    break
                            if not found:
                                clusters.append([(cx, cy)])
                        person_count = len(clusters)

                violations = detections[detections['name'].isin(violation_classes)]
                ppe_detected = detections[detections['name'].isin(ppe_classes)]
                violation_list = violations['name'].tolist()
                violation_count = len(violation_list)

                def violation_label(v):
                    labels = {
                        'no-helmet':  '⛑ No Helmet',
                        'no-vest':    '🦺 No Safety Vest',
                        'no-gloves':  '🧤 No Gloves',
                        'no-goggles': '🥽 No Goggles',
                        'no-boots':   '👢 No Safety Boots'
                    }
                    return labels.get(v, v)

                annotated = results.render()[0]
                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 50])
                b64_frame = base64.b64encode(buffer).decode('utf-8')

                self.socketio.emit('frame', {
                    'cam_id': self.cam_id,
                    'frame': b64_frame,
                    'person_count': person_count,
                    'violations': violation_list,
                    'violation_count': violation_count,
                    'ppe_detected': ppe_detected['name'].tolist()
                })

                if violation_list:
                    current_time = time.time()
                    new_violations = []

                    for v in set(violation_list):
                        if current_time - self.alert_cooldown.get(v, 0) > 30:
                            new_violations.append(v)
                            self.alert_cooldown[v] = current_time

                    if new_violations:
                        details = [violation_label(v) for v in new_violations]
                        self.socketio.emit('alert', {
                            'cam_id':       self.cam_id,
                            'violations':   new_violations,
                            'details':      details,
                            'worker_count': person_count,
                            'timestamp':    time.strftime('%H:%M:%S')
                        })
                else:
                    self.alert_cooldown = {}

                self.socketio.sleep(0.01)

            print(f'Stream loop ended for camera {self.cam_id}')
        except Exception as e:
            print(f"Error in stream loop for {self.cam_id}: {e}")
        finally:
            self.running = False
            self._notify_stopped()
            if self.cap:
                self.cap.release()
                self.cap = None
            print(f'Stream loop cleanup finished for camera {self.cam_id}')

    def _notify_stopped(self):
    
        pass


class CameraManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.cameras = {}
        self.streams_running = set()
        self.url_map = {}       # { cam_id: original_source_url }
        self.id_counter = 1

        print('Loading YOLOv5 model...')
        self.model = torch.hub.load(
            os.path.join(os.path.dirname(__file__), 'yolov5'),
            'custom',
            path=os.path.join(os.path.dirname(__file__), 'models', 'best.pt'),
            source='local'
        )
        self.model.conf = 0.43
        self.model.iou = 0.45
        print('Model loaded.')

    def add_camera(self, source):
        if source == 0 or source == "0":
            cam_id = 0
        else:
            cam_id = self.id_counter
            self.id_counter += 1

        if cam_id not in self.cameras:
            # (source, cam_id, model, socketio)
            stream = CameraStream(source, cam_id, self.model, self.socketio)

            # Patch _notify_stopped so the loop can clean up streams_running
            def make_stopper(cid):
                def _stopper():
                    self.streams_running.discard(cid)
                return _stopper
            stream._notify_stopped = make_stopper(cam_id)

            self.cameras[cam_id] = stream
            self.url_map[cam_id] = source
            print(f'Camera {cam_id} added for source {source}')
        return cam_id

    def remove_camera(self, cam_id):
        print(f"DEBUG: Trying to remove {cam_id} (type: {type(cam_id)})")
        print(f"DEBUG: Current keys: {list(self.cameras.keys())}")
        if cam_id in self.cameras:
            self.cameras[cam_id].stop()
            del self.cameras[cam_id]
            self.streams_running.discard(cam_id)
            self.url_map.pop(cam_id, None)
            print(f"Camera {cam_id} removed successfully")
        else:
            print(f"DEBUG: Camera {cam_id} NOT FOUND in dictionary!")

    def start_stream(self, cam_id):
        if cam_id not in self.cameras:
            print(f'Camera {cam_id} not found')
            return

        if cam_id in self.streams_running:
            print(f'Stream already running for cam {cam_id}')
            return

        success = self.cameras[cam_id].open_capture()
        if not success:
            print(f'Failed to open camera {cam_id}')
            return

        self.streams_running.add(cam_id)
        self.socketio.start_background_task(self.cameras[cam_id]._loop)
        print(f'Background task started for cam {cam_id}')

    def stop_stream(self, cam_id):
        if cam_id in self.cameras:
            self.cameras[cam_id].stop()
            self.streams_running.discard(cam_id)