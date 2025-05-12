from flask import Flask, request, jsonify
import numpy as np
import noisereduce as nr
import heartpy as hp
from scipy.signal import butter, filtfilt
import json
from datetime import datetime
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'ppg_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process_ppg():
    try:
        # Save incoming data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_path = os.path.join(UPLOAD_FOLDER, f"ppg_compact_{timestamp}.json")
        
        data = request.get_json()
        with open(upload_path, 'w') as f:
            json.dump(data, f, indent=4)

        # Process data
        timestamps = [float(frame['t']) for frame in data]
        red_values = [float(frame['r']) for frame in data]

        # Signal processing
        signal = np.array(red_values)
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        reduced_noise = nr.reduce_noise(
            y=signal,
            sr=30,
            stationary=True,
            prop_decrease=0.95
        )
        
        filtered = butter_bandpass_filter(
            reduced_noise,
            lowcut=0.7,
            highcut=4.0,
            fs=30
        )
        
        working_data, measures = hp.process(filtered, 30)

        # Create analysis JSON
        analysis_data = {
            "timestamp": timestamp,
            "bpm": measures['bpm'],
            "hrv": measures['rmssd'],
            "peaks": working_data['peaklist'],
            "processed_frames": len(data),
            "mean_frame_time": np.mean(np.diff(timestamps))
        }
        
        analysis_path = os.path.join(UPLOAD_FOLDER, f"ppg_analysis_{timestamp}.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=4)

        return jsonify({
            "status": "success",
            "analysis_path": analysis_path,
            **analysis_data
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
