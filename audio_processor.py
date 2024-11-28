import librosa
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

class AudioProcessor:
    def __init__(self, sampling_rate=16000, mel_step_size=16, mel_window_size=800, mel_channels=80):
        self.sampling_rate = sampling_rate
        self.mel_step_size = mel_step_size
        self.mel_window_size = mel_window_size
        self.mel_channels = mel_channels

    def load_audio(self, audio_path):
        """Carrega e normaliza o áudio."""
        audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
        
        # Normalizar o áudio
        audio = audio / np.abs(audio).max()
        
        return audio, sr

    def extract_mel_features(self, audio_path):
        """Extrai características mel-spectrogram do áudio."""
        # Carregar áudio
        audio, _ = self.load_audio(audio_path)
        
        # Calcular mel spectrogram
        mel_basis = librosa.filters.mel(
            sr=self.sampling_rate,
            n_fft=self.mel_window_size,
            n_mels=self.mel_channels
        )
        
        # Calcular STFT
        stft = librosa.core.stft(
            y=audio,
            n_fft=self.mel_window_size,
            hop_length=self.mel_step_size,
            win_length=self.mel_window_size
        )
        
        # Converter para mel scale
        mel = np.dot(mel_basis, np.abs(stft))
        
        # Converter para dB
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        
        # Normalizar
        mel = (mel - mel.mean()) / mel.std()
        
        return mel

    def align_audio_to_video(self, mel_features, video_frames):
        """Alinha as características do áudio com os frames do vídeo."""
        # Obter número de frames de áudio e vídeo
        n_audio_frames = mel_features.shape[1]
        n_video_frames = len(video_frames)
        
        # Criar função de interpolação
        x = np.linspace(0, n_audio_frames - 1, n_audio_frames)
        x_new = np.linspace(0, n_audio_frames - 1, n_video_frames)
        
        # Interpolar para cada canal mel
        aligned_features = []
        for channel in range(mel_features.shape[0]):
            interpolator = interp1d(x, mel_features[channel], kind='linear')
            aligned_features.append(interpolator(x_new))
        
        return np.array(aligned_features)

    def preprocess_audio(self, audio_path, video_frames=None):
        """Processa o áudio completo para uso no modelo."""
        # Extrair características mel
        mel_features = self.extract_mel_features(audio_path)
        
        # Alinhar com o vídeo se necessário
        if video_frames is not None:
            mel_features = self.align_audio_to_video(mel_features, video_frames)
        
        # Converter para tensor
        mel_features = tf.convert_to_tensor(mel_features, dtype=tf.float32)
        
        # Adicionar dimensão de batch
        mel_features = tf.expand_dims(mel_features, 0)
        
        return mel_features
