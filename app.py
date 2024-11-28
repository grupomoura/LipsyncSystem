import os
from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
import face_recognition
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
from lipsync_processor import LipSyncProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['STATIC_FOLDER'] = 'static'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mp3', 'wav'}

# Inicializar o processador de lipsync
lipsync_processor = LipSyncProcessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) == 0:
        return None, "Nenhum rosto detectado na imagem/vídeo."
    
    faces = []
    for idx, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        
        # Salvar imagem do rosto para visualização
        face_path = os.path.join(app.config['STATIC_FOLDER'], f'face_{idx}.jpg')
        cv2.imwrite(face_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        
        faces.append({
            'id': idx,
            'location': face_location,
            'preview_url': f'/static/face_{idx}.jpg'
        })
    
    return faces, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files and 'image' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'error': 'Arquivo de áudio não enviado'}), 400

    # Handle video/image upload
    media_file = request.files.get('video') or request.files.get('image')
    audio_file = request.files['audio']

    if media_file.filename == '' or audio_file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    if not (allowed_file(media_file.filename) and allowed_file(audio_file.filename)):
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

    # Save files
    media_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(media_file.filename))
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
    
    media_file.save(media_path)
    audio_file.save(audio_path)

    # Detect faces
    faces, error = detect_faces(media_path)
    if error:
        return jsonify({'error': error}), 400

    if len(faces) > 1:
        # Return face options to frontend
        return jsonify({
            'multiple_faces': True,
            'faces': faces,
            'media_path': media_path,
            'audio_path': audio_path
        })

    # Process single face
    try:
        result_path = lipsync_processor.process_media(media_path, audio_path, 0)
        return jsonify({
            'success': True,
            'result_path': f'/static/{os.path.basename(result_path)}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    media_path = data.get('media_path')
    audio_path = data.get('audio_path')
    face_id = data.get('face_id')
    
    try:
        result_path = lipsync_processor.process_media(media_path, audio_path, face_id)
        # Mover o resultado para a pasta static
        static_result_path = os.path.join(app.config['STATIC_FOLDER'], os.path.basename(result_path))
        os.rename(result_path, static_result_path)
        
        return jsonify({
            'success': True,
            'result_path': f'/static/{os.path.basename(result_path)}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/face/<int:face_id>')
def get_face(face_id):
    face_path = os.path.join(app.config['STATIC_FOLDER'], f'face_{face_id}.jpg')
    return send_file(face_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
