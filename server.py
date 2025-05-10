from flask import Flask, request, jsonify, send_from_directory
import os
import tempfile
from werkzeug.utils import secure_filename
from ppgextractor import process_video  # Your existing extraction code

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
OUTPUT_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "No video file"}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        # Use your existing process_video function
        result = process_video(video_path)
        
        if not result or 'graph_path' not in result:
            raise ValueError("Processing failed")
            
        return jsonify({
            "status": "success",
            "graph_url": f"/static/{os.path.basename(result['graph_path'])}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "PPG Processing API"})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
