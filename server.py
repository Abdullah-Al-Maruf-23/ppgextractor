import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from io import BytesIO
import uuid
import tempfile
import noisereduce as nr
import heartpy as hp

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
OUTPUT_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def extract_ppg(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ROI parameters (adjust based on your video)
    roi_x, roi_y, roi_width, roi_height = 100, 50, 100, 50

    red_channel = []
    timestamps = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV color space for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for skin color
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply mask to original frame
        skin = cv2.bitwise_and(frame, frame, mask=mask)

        # Extract ROI from forehead area
        roi = skin[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

        if roi.size > 0:
            red_mean = np.mean(roi[:,:,2])  # Red channel
            red_channel.append(red_mean)
            timestamps.append(frame_count / fps)

        frame_count += 1

    cap.release()

    if len(red_channel) < 10:  # Minimum frames needed
        return None, None, None

    # Convert to numpy arrays
    ppg_signal = np.array(red_channel)
    timestamps = np.array(timestamps)

    # Normalize signal
    ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)

    # Noise reduction
    reduced_noise = nr.reduce_noise(
        y=ppg_signal,
        sr=fps,
        stationary=True,
        prop_decrease=0.95
    )

    # Bandpass filter (0.7Hz to 4Hz = 42-240 bpm)
    filtered = butter_bandpass_filter(
        reduced_noise,
        lowcut=0.7,
        highcut=4.0,
        fs=fps
    )

    # Peak detection using HeartPy
    working_data, measures = hp.process(filtered, fps)

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Raw signal
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, ppg_signal, 'r', alpha=0.5, label='Raw PPG')
    plt.title('PPG Signal Processing Pipeline')
    plt.ylabel('Amplitude')
    plt.legend()

    # Filtered signal
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, filtered, 'b', label='Filtered PPG')
    plt.ylabel('Amplitude')
    plt.legend()

    # With detected peaks
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, filtered, 'b', label='Filtered PPG')
    plt.plot(
        np.array(working_data['peaklist'])/fps, 
        filtered[working_data['peaklist']], 
        'ro', label='Detected Peaks'
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()

    # Save plot
    graph_filename = f"ppg_{uuid.uuid4().hex}.png"
    graph_path = os.path.join(app.config['OUTPUT_FOLDER'], graph_filename)
    plt.savefig(graph_path, dpi=120)
    plt.close()

    return graph_filename, working_data, measures

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
            graph_filename, working_data, measures = extract_ppg(video_path)
            if not graph_filename:
                return jsonify({"error": "PPG extraction failed"}), 500

            return jsonify({
                "status": "success",
                "graph_url": f"{request.host_url}static/{graph_filename}",
                "heart_rate": measures['bpm'],
                "hrv": measures['rmssd'],
                "peaks": working_data['peaklist']
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up video file
            if os.path.exists(video_path):
                os.remove(video_path)

    return jsonify({"error": "Invalid file"}), 400

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "PPG Processing API"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
