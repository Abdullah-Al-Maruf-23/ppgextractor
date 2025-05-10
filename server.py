import os
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'mp4'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_ppg(video_path):
    cap = cv2.VideoCapture(video_path)
    red_values = []
    timestamps = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        timestamps.append(current_time)
        
        # Simple PPG extraction (adjust ROI as needed)
        forehead_roi = frame[50:100, 100:200] if frame is not None else None
        if forehead_roi is not None:
            red_mean = np.mean(forehead_roi[:,:,2])  # Red channel
            red_values.append(red_mean)
    
    cap.release()
    
    if not red_values:
        return None
    
    # Process signal
    red_values = (red_values - np.mean(red_values)) / np.std(red_values)
    
    # Create plot
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps[:len(red_values)], red_values, 'r')
    plt.title('PPG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Intensity')
    plt.grid(True)
    
    # Save plot
    graph_filename = f"ppg_{uuid.uuid4().hex}.png"
    graph_path = os.path.join(app.config['OUTPUT_FOLDER'], graph_filename)
    plt.savefig(graph_path)
    plt.close()
    
    return graph_filename

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "No video file"}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        try:
            graph_filename = extract_ppg(video_path)
            if not graph_filename:
                return jsonify({"error": "PPG extraction failed"}), 500
                
            return jsonify({
                "status": "success",
                "graph_url": f"{request.host_url}static/{graph_filename}"
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "Invalid file"}), 400

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
