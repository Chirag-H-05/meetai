"""
Flask API Server for Audio Analysis Pipeline
Supports large audio uploads + automatic WAV conversion
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from pathlib import Path
import json
import threading
import logging
from werkzeug.utils import secure_filename
import subprocess
import uuid
import os

# -------------------------
# Flask App Setup
# -------------------------
app = Flask(__name__, static_folder='.')
CORS(app)

# Allow very large uploads (5 GB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  

# -------------------------
# Paths
# -------------------------
UPLOAD_FOLDER = Path('uploads')
OUTPUTS_FOLDER = Path('outputs')
REPORTS_FOLDER = Path('reports')

for folder in [UPLOAD_FOLDER, OUTPUTS_FOLDER, REPORTS_FOLDER]:
    folder.mkdir(exist_ok=True)

# -------------------------
# Config
# -------------------------
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg'}

current_pipeline = None
pipeline_thread = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Helpers
# -------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_wav(input_path: Path) -> Path:
    """
    Convert audio to 16kHz mono WAV using FFmpeg (Windows-safe)
    """
    output_wav = OUTPUTS_FOLDER / f"{input_path.stem}_{uuid.uuid4().hex}.wav"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(output_wav)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False
    )

    if result.returncode != 0:
        logger.error("FFmpeg conversion failed")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(result.stderr)

    if not output_wav.exists():
        raise RuntimeError("FFmpeg finished but WAV file not created")

    return output_wav

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return send_file('web_ui.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload large audio files and convert to WAV
    """
    logger.info(f"Received file: {file.filename}")

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    input_path = UPLOAD_FOLDER / filename

    # Save file safely (streaming)
    file.save(input_path)

    try:
        wav_path = convert_to_wav(input_path)

        return jsonify({
            'success': True,
            'original_file': filename,
            'wav_file': wav_path.name,
            'wav_path': str(wav_path)
        })

    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return jsonify({'error': 'Audio conversion failed'}), 500


@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    """
    Start analysis pipeline on WAV file
    """
    global current_pipeline, pipeline_thread

    data = request.json or {}
    wav_file = data.get('wav_file')

    if not wav_file:
        return jsonify({'error': 'No WAV file specified'}), 400

    audio_path = OUTPUTS_FOLDER / wav_file

    if not audio_path.exists():
        return jsonify({'error': 'WAV file not found'}), 404

    # Example config (extend later)
    config = {
        'base_dir': '.',
        'ollama_path': data.get(
            'ollama_path',
            r"C:/Users/ISFL-RT000268/AppData/Local/Programs/Ollama/ollama.exe"
        )
    }

    # -------------------------
    # Placeholder pipeline logic
    # -------------------------
    status = {
        'stage': 'starting',
        'progress': 0,
        'message': 'Analysis started',
        'errors': []
    }

    status_file = OUTPUTS_FOLDER / 'pipeline_status.json'
    with open(status_file, 'w') as f:
        json.dump(status, f)

    return jsonify({
        'success': True,
        'message': 'Analysis started',
        'wav_file': wav_file
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    status_file = OUTPUTS_FOLDER / 'pipeline_status.json'

    if not status_file.exists():
        return jsonify({
            'stage': 'idle',
            'progress': 0,
            'message': 'No analysis running',
            'errors': []
        })

    with open(status_file, 'r') as f:
        return jsonify(json.load(f))


@app.route('/api/results', methods=['GET'])
def get_results():
    viz_file = OUTPUTS_FOLDER / 'visualization_data.json'

    if not viz_file.exists():
        return jsonify({'error': 'No results available'}), 404

    with open(viz_file, 'r') as f:
        return jsonify(json.load(f))


@app.route('/api/report/<report_name>', methods=['GET'])
def get_report(report_name):
    report_path = REPORTS_FOLDER / report_name

    if not report_path.exists():
        return jsonify({'error': 'Report not found'}), 404

    return send_file(report_path, as_attachment=False)


@app.route('/api/reports/list', methods=['GET'])
def list_reports():
    reports = []

    for report_file in REPORTS_FOLDER.glob('*'):
        if report_file.is_file():
            reports.append({
                'name': report_file.name,
                'size': report_file.stat().st_size,
                'url': f'/api/report/{report_file.name}'
            })

    return jsonify(reports)


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUTS_FOLDER, filename)


@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(REPORTS_FOLDER, filename)


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'uploads_dir': str(UPLOAD_FOLDER.resolve()),
        'outputs_dir': str(OUTPUTS_FOLDER.resolve()),
        'reports_dir': str(REPORTS_FOLDER.resolve())
    })


# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("AUDIO ANALYSIS API SERVER")
    print("=" * 60)
    print("✅ Supports MP3 / WAV / FLAC / M4A / OGG")
    print("✅ Auto converts to 16kHz mono WAV")
    print("✅ Large uploads supported (GBs)")
    print("\n🌐 http://localhost:5050\n")

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5050
    )
