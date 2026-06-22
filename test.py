import torch
import cv2
import sys
import os
import time

# ── config ──────────────────────────────────────────────────
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
CONF        = 0.43
IOU         = 0.45
IMG_SIZE    = 320

VIOLATION_CLASSES = ['no-boots','no-gloves','no-goggles','no-helmet','no-vest']
PPE_CLASSES       = ['boots','gloves','goggles','helmet','vest']
# ────────────────────────────────────────────────────────────

sys.path.insert(0, YOLOV5_PATH)

print('Loading model...')
model = torch.hub.load(YOLOV5_PATH, 'custom', path=MODEL_PATH, source='local')
model.conf    = CONF
model.iou     = IOU
model.imgsz   = IMG_SIZE
print('Model loaded.\n')


def draw_boxes(frame, detections):
    for _, row in detections.iterrows():
        x1,y1,x2,y2 = int(row.xmin),int(row.ymin),int(row.xmax),int(row.ymax)
        label = row['name']
        conf  = row['confidence']
        color = (0,0,255) if label in VIOLATION_CLASSES else (0,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}',
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)
    return frame


def print_summary(detections, elapsed):
    print(f'\n  Inference time : {elapsed*1000:.1f} ms')
    print(f'  Total detections: {len(detections)}')
    if len(detections):
        for _, row in detections.iterrows():
            tag = '⚠ VIOLATION' if row['name'] in VIOLATION_CLASSES else '✓ PPE OK  '
            print(f'    {tag}  {row["name"]:15s}  conf={row["confidence"]:.2f}')
    violations = detections[detections['name'].isin(VIOLATION_CLASSES)]
    if len(violations):
        print(f'\n  🚨 VIOLATIONS DETECTED: {", ".join(violations["name"].tolist())}')
    else:
        print('\n  ✅ No violations detected')


def test_image(path):
    print(f'\n{"="*50}')
    print(f'Testing image: {path}')
    frame = cv2.imread(path)
    if frame is None:
        print('ERROR: Could not read image')
        return

    t0 = time.time()
    results = model(frame)
    elapsed = time.time() - t0

    detections = results.pandas().xyxy[0]
    print_summary(detections, elapsed)

    annotated = draw_boxes(frame.copy(), detections)
    cv2.imshow(f'Result - {os.path.basename(path)}', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    out_path = path.replace('.jpg','_result.jpg').replace('.png','_result.png')
    cv2.imwrite(out_path, annotated)
    print(f'  Saved: {out_path}')


def test_video(path):
    print(f'\n{"="*50}')
    print(f'Testing video: {path}')
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print('ERROR: Could not open video')
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'  Resolution: {width}x{height}  FPS: {fps:.1f}  Frames: {total}')

    out_path = path.replace('.mp4','_result.mp4')
    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))

    frame_num   = 0
    skip        = 2          # run YOLO every 2nd frame
    last_dets   = None

    print('  Processing... press Q to quit early\n')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        if frame_num % skip == 0:
            t0      = time.time()
            results = model(small)
            elapsed = time.time() - t0
            last_dets = results.pandas().xyxy[0]

            violations = last_dets[last_dets['name'].isin(VIOLATION_CLASSES)]
            status = f'Frame {frame_num}/{total} | {elapsed*1000:.0f}ms'
            if len(violations):
                status += f' | ⚠ {", ".join(violations["name"].tolist())}'
            print(f'\r  {status}', end='')

        if last_dets is not None:
            annotated = draw_boxes(frame.copy(), last_dets)
        else:
            annotated = frame

        writer.write(annotated)
        cv2.imshow('PPE Detection Test', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f'\n\n  Done! Saved result to: {out_path}')


def test_webcam():
    print(f'\n{"="*50}')
    print('Testing webcam (press Q to quit)...')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Could not open webcam')
        return

    frame_num = 0
    last_dets = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        if frame_num % 3 == 0:
            results  = model(small)
            last_dets = results.pandas().xyxy[0]
            violations = last_dets[last_dets['name'].isin(VIOLATION_CLASSES)]
            if len(violations):
                print(f'\r  ⚠ {", ".join(violations["name"].tolist())}     ', end='')
            else:
                print(f'\r  ✅ No violations                              ', end='')

        if last_dets is not None:
            annotated = draw_boxes(frame.copy(), last_dets)
        else:
            annotated = frame

        cv2.imshow('PPE Detection - Webcam Test', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── MAIN ────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\nPPE Detection Model Test')
    print('========================')
    print('1. Test image')
    print('2. Test video file')
    print('3. Test webcam live')
    choice = input('\nChoose (1/2/3): ').strip()

    if choice == '1':
        path = input('Image path (e.g. test.jpg): ').strip().strip('"')
        test_image(path)
    elif choice == '2':
        path = input('Video path (e.g. test.mp4): ').strip().strip('"')
        test_video(path)
    elif choice == '3':
        test_webcam()
    else:
        print('Invalid choice')