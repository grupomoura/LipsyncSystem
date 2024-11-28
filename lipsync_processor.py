import cv2
import numpy as np
import tensorflow as tf
import face_recognition
from moviepy.editor import VideoFileClip, AudioFileClip
import os
import tempfile
from PIL import Image
from audio_processor import AudioProcessor
from wav2lip_model import Wav2LipModel

class LipSyncProcessor:
    def __init__(self, model_path=None):
        # Inicializar processadores
        self.audio_processor = AudioProcessor()
        self.model = Wav2LipModel()
        
        # Carregar pesos do modelo se fornecidos
        if model_path:
            self.model.load_weights(model_path)
        
    def _load_model(self):
        """Carrega o modelo Wav2Lip."""
        return self.model
        
    def _extract_audio_features(self, audio_path):
        """Extrai características do áudio para o modelo Wav2Lip."""
        return self.audio_processor.preprocess_audio(audio_path)

    def _get_face_region(self, frame, face_location):
        """Extrai a região do rosto da imagem."""
        top, right, bottom, left = face_location
        face_region = frame[top:bottom, left:right]
        
        # Converter para RGB se necessário
        if len(face_region.shape) == 2:
            face_region = cv2.cvtColor(face_region, cv2.COLOR_GRAY2RGB)
        elif face_region.shape[2] == 4:
            face_region = cv2.cvtColor(face_region, cv2.COLOR_RGBA2RGB)
            
        return face_region

    def _apply_lipsync(self, face_region, mel_features):
        """Aplica o lipsync em uma região do rosto."""
        # Redimensionar face para 96x96 (tamanho esperado pelo modelo)
        face_tensor = tf.image.resize(face_region, (96, 96))
        
        # Gerar face sincronizada
        synced_face = self.model.predict(face_tensor, mel_features)
        
        # Redimensionar de volta ao tamanho original
        synced_face = tf.image.resize(synced_face, (face_region.shape[0], face_region.shape[1]))
        
        return synced_face.numpy().astype(np.uint8)

    def _blend_face(self, original_frame, new_face_region, face_location):
        """Mistura o rosto processado de volta no frame original."""
        top, right, bottom, left = face_location
        result = original_frame.copy()
        
        # Criar máscara para mesclagem suave
        mask = np.zeros((bottom-top, right-left), dtype=np.float32)
        cv2.ellipse(mask, 
                   center=(mask.shape[1]//2, mask.shape[0]//2),
                   axes=(mask.shape[1]//3, mask.shape[0]//2),
                   angle=0, startAngle=0, endAngle=360,
                   color=1, thickness=-1)
        
        # Aplicar feather à máscara
        mask = cv2.GaussianBlur(mask, (19, 19), 0)
        
        # Expandir máscara para 3 canais
        mask = np.expand_dims(mask, axis=-1)
        
        # Mesclar faces usando a máscara
        result[top:bottom, left:right] = (
            new_face_region * mask + 
            original_frame[top:bottom, left:right] * (1 - mask)
        ).astype(np.uint8)
        
        return result

    def process_video(self, video_path, audio_path, face_id=0):
        """Processa um vídeo com lipsync."""
        # Carregar vídeo e áudio
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        # Extrair características do áudio
        mel_features = self._extract_audio_features(audio_path)
        
        # Criar arquivo temporário para o resultado
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            output_path = temp_output.name
        
        # Processar cada frame
        processed_frames = []
        for frame in video.iter_frames():
            # Detectar rostos
            face_locations = face_recognition.face_locations(frame)
            
            if len(face_locations) > face_id:
                face_location = face_locations[face_id]
                face_region = self._get_face_region(frame, face_location)
                
                # Aplicar lipsync
                new_face_region = self._apply_lipsync(face_region, mel_features)
                
                # Misturar o rosto processado de volta no frame
                processed_frame = self._blend_face(frame, new_face_region, face_location)
                processed_frames.append(processed_frame)
            else:
                processed_frames.append(frame)
        
        # Criar vídeo final
        fps = video.fps if video.fps else 30
        output_video = VideoFileClip(processed_frames, fps=fps)
        final_video = output_video.set_audio(audio)
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        return output_path

    def process_image(self, image_path, audio_path, face_id=0):
        """Processa uma imagem estática com lipsync."""
        # Carregar imagem
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar rostos
        face_locations = face_recognition.face_locations(image_rgb)
        
        if len(face_locations) <= face_id:
            raise ValueError("Face ID não encontrado na imagem")
        
        # Extrair características do áudio
        mel_features = self._extract_audio_features(audio_path)
        
        # Processar frames
        face_location = face_locations[face_id]
        face_region = self._get_face_region(image_rgb, face_location)
        
        # Criar frames para o vídeo
        processed_frames = []
        audio_duration = AudioFileClip(audio_path).duration
        n_frames = int(audio_duration * 30)  # 30 fps
        
        for i in range(n_frames):
            # Aplicar lipsync com o frame atual do áudio
            current_mel = mel_features[:, :, i:i+1, :]
            new_face_region = self._apply_lipsync(face_region, current_mel)
            
            # Misturar o rosto processado de volta na imagem
            processed_frame = self._blend_face(image_rgb, new_face_region, face_location)
            processed_frames.append(processed_frame)
        
        # Criar arquivo temporário para o resultado
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            output_path = temp_output.name
        
        # Criar vídeo final
        output_video = VideoFileClip(processed_frames, fps=30)
        audio = AudioFileClip(audio_path)
        final_video = output_video.set_audio(audio)
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        return output_path

    def process_media(self, media_path, audio_path, face_id=0):
        """Processa mídia (vídeo ou imagem) com lipsync."""
        if media_path.lower().endswith(('.mp4')):
            return self.process_video(media_path, audio_path, face_id)
        elif media_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return self.process_image(media_path, audio_path, face_id)
        else:
            raise ValueError("Formato de arquivo não suportado")
