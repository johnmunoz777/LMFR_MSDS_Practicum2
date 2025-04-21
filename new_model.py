import base64
import sqlite3
from collections import defaultdict

import cv2
import cvzone
import numpy as np
from flask import Flask, request, jsonify, Response
from ultralytics import YOLO
import torch

app = Flask(__name__)

# Check GPU availability
if torch.cuda.is_available():
    print("GPU is available, using GPU for inference.")
else:
    print("GPU is NOT available, using CPU for inference.")

# Load YOLO model once at startup
model = YOLO("newest_yolo.pt")

# Define class names and field names (15 fields)
classNames = [
    'angela', 'classmate', 'giuliana', 'javier', 'john',
    'maite', 'mike', 'ron', 'shanti', 'tom', 'vilma', 'will','kevin','shirley'
]
FIELD_NAMES = [
    "ID", "Name", "Age", "Date of Birth", "Address", "Loyalty",
    "Member Since", "Gender", "Email", "Phone Number", "Membership Type",
    "Status", "Occupation", "Interests", "Marital Status"
]

# Drawing parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 2
LINE_HEIGHT = 30
FONT_COLOR = (0, 255, 0)  # Green

# ----------------------- Helper Functions ----------------------- #

def getProfile(member_id, db_path='capstone.db'):
    """Retrieve a profile from the SQLite database given a member_id."""
    conn = sqlite3.connect(db_path)
    cmd = "SELECT * FROM ski WHERE id=" + str(member_id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

def draw_profile_info(img, profile, x1, y1, w, h, field_names, font, font_scale, thickness, line_height, font_color):
    """
    Draw the profile details below the detection bounding box.
    This writes all the profile fields on the image.
    """
    startY = y1 + h + 20
    for i, field_name in enumerate(field_names):
        if i < len(profile):
            text = f"{field_name}: {profile[i]}"
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(img,
                          (x1, startY + i * line_height - text_h - 5),
                          (x1 + text_w, startY + i * line_height + 5),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(img, text,
                        (x1, startY + i * line_height),
                        font, font_scale, font_color, thickness)

def process_frame(img, model, conf_threshold, class_names, field_names, font, font_scale, thickness, line_height, font_color):
    """
    Process one image frame:
      - Runs detection with YOLO.
      - Overlays bounding boxes and profile details on the image.
      - Builds a list of detection dictionaries with all details.
    
    Returns:
      annotated_img: The image with annotations.
      detections: A list of dictionaries with detection data.
    """
    detections = []
    results = model(img)  # Using non-stream mode for efficiency
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cls = int(box.cls[0])
            profile = getProfile(cls + 1)
            
            # Build a dictionary of profile details if available
            profile_dict = {}
            if profile is not None:
                for i, field in enumerate(field_names):
                    if i < len(profile):
                        profile_dict[field] = profile[i]
                detected_name = profile[1]
                detected_status = profile[11] if len(profile) > 11 else "Inactive"
            else:
                detected_name = class_names[cls] if 0 <= cls < len(class_names) else "Unknown"
                detected_status = "Inactive"

            conf_percent = int(conf * 100)
            label_text = f'{detected_name} - {detected_status} {conf_percent}%'
            # Choose bounding box color: green if active, red otherwise
            box_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
            cvzone.putTextRect(img, label_text, (max(0, x1), max(35, y1)),
                               scale=2, thickness=2, colorR=box_color)
            if profile is not None:
                draw_profile_info(img, profile, x1, y1, w, h,
                                  field_names, font, font_scale, thickness, line_height, font_color)
            
            detections.append({
                'name': detected_name,
                'status': detected_status,
                'confidence': conf,
                'box': [x1, y1, x2, y2],
                'label': label_text,
                'profile': profile_dict
            })
    return img, detections

def convert_image_to_base64(img):
    """
    Convert an OpenCV image to a base64-encoded string.
    We reduce JPEG quality to 80 for faster encoding and smaller size.
    """
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# ----------------------- Webcam Streaming ----------------------- #

def generate():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use DirectShow backend
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_img, detections = process_frame(frame, model, 0.7, classNames,
                                                  FIELD_NAMES, FONT, FONT_SCALE, THICKNESS,
                                                  LINE_HEIGHT, FONT_COLOR)
        _, buffer = cv2.imencode('.jpg', annotated_img)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ----------------------- Flask Endpoint ----------------------- #

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects an image file (captured from a webcam) in the POST request.
    Processes the image with YOLO, annotates it (drawing bounding boxes, colored status, name, and all 15 fields),
    and returns:
      - A list of detection dictionaries with all the details.
      - The annotated image as a base64-encoded string.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    img_bytes = file.read()
    # Decode image bytes directly using OpenCV
    nparr = np.frombuffer(img_bytes, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    annotated_img, detections = process_frame(cv_img, model, 0.7, classNames,
                                              FIELD_NAMES, FONT, FONT_SCALE, THICKNESS,
                                              LINE_HEIGHT, FONT_COLOR)
    annotated_img_b64 = convert_image_to_base64(annotated_img)
    return jsonify({
        'predictions': detections,
        'annotated_image': annotated_img_b64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
