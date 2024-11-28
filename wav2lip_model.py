import tensorflow as tf
from tensorflow.keras import layers, Model

class Wav2LipModel:
    def __init__(self):
        self.face_encoder = self._build_face_encoder()
        self.audio_encoder = self._build_audio_encoder()
        self.generator = self._build_generator()
        self.model = self._build_complete_model()

    def _build_face_encoder(self):
        """Constrói o encoder para características faciais."""
        inputs = layers.Input(shape=(96, 96, 3))
        
        # Encoder
        x = layers.Conv2D(32, 3, strides=1, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        return Model(inputs=inputs, outputs=x, name='face_encoder')

    def _build_audio_encoder(self):
        """Constrói o encoder para características de áudio."""
        inputs = layers.Input(shape=(80, None, 1))  # mel features
        
        # Encoder
        x = layers.Conv2D(32, 3, strides=1, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(64, 3, strides=(2, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(128, 3, strides=(2, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        return Model(inputs=inputs, outputs=x, name='audio_encoder')

    def _build_generator(self):
        """Constrói o gerador que combina características de áudio e vídeo."""
        face_inputs = layers.Input(shape=(12, 12, 256))  # encoded face
        audio_inputs = layers.Input(shape=(20, None, 128))  # encoded audio
        
        # Processar áudio
        x_audio = layers.Conv2D(256, 3, strides=1, padding='same')(audio_inputs)
        x_audio = layers.BatchNormalization()(x_audio)
        x_audio = layers.ReLU()(x_audio)
        
        # Concatenar face e áudio
        x = layers.Concatenate()([face_inputs, x_audio])
        
        # Decoder
        x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Saída final
        outputs = layers.Conv2D(3, 1, activation='tanh')(x)
        
        return Model(inputs=[face_inputs, audio_inputs], outputs=outputs, name='generator')

    def _build_complete_model(self):
        """Constrói o modelo completo Wav2Lip."""
        face_inputs = layers.Input(shape=(96, 96, 3))
        audio_inputs = layers.Input(shape=(80, None, 1))
        
        # Codificar face e áudio
        encoded_face = self.face_encoder(face_inputs)
        encoded_audio = self.audio_encoder(audio_inputs)
        
        # Gerar face sincronizada
        synced_face = self.generator([encoded_face, encoded_audio])
        
        return Model(inputs=[face_inputs, audio_inputs], outputs=synced_face, name='wav2lip')

    def load_weights(self, weights_path):
        """Carrega os pesos do modelo."""
        self.model.load_weights(weights_path)

    def predict(self, face_frame, mel_features):
        """Gera um frame sincronizado."""
        # Preprocessar entrada
        face_frame = tf.image.resize(face_frame, (96, 96))
        face_frame = (face_frame / 127.5) - 1.0  # Normalizar para [-1, 1]
        
        # Adicionar dimensão de batch se necessário
        if len(face_frame.shape) == 3:
            face_frame = tf.expand_dims(face_frame, 0)
        
        # Fazer predição
        synced_frame = self.model.predict([face_frame, mel_features])
        
        # Pós-processar saída
        synced_frame = (synced_frame + 1.0) * 127.5
        synced_frame = tf.clip_by_value(synced_frame, 0, 255)
        
        return synced_frame[0]  # Remover dimensão de batch
