#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import logging
from lipsync_processor import LipSyncProcessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessorCommands:
    def __init__(self):
        self.processor = None
        self.weights_path = os.path.join('weights', 'wav2lip_gan.pth')
        
    def initialize_processor(self):
        """Inicializa o processador com os pesos do modelo."""
        if not os.path.exists(self.weights_path):
            logger.error(f"Pesos do modelo não encontrados em {self.weights_path}")
            logger.info("Execute 'python download_weights.py' primeiro para baixar os pesos")
            sys.exit(1)
            
        try:
            self.processor = LipSyncProcessor(model_path=self.weights_path)
            logger.info("Processador inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar o processador: {str(e)}")
            sys.exit(1)
    
    def validate_files(self, media_path: str, audio_path: str) -> bool:
        """Valida os arquivos de entrada."""
        if not os.path.exists(media_path):
            logger.error(f"Arquivo de mídia não encontrado: {media_path}")
            return False
            
        if not os.path.exists(audio_path):
            logger.error(f"Arquivo de áudio não encontrado: {audio_path}")
            return False
            
        media_ext = os.path.splitext(media_path)[1].lower()
        audio_ext = os.path.splitext(audio_path)[1].lower()
        
        valid_media = media_ext in ['.mp4', '.png', '.jpg', '.jpeg']
        valid_audio = audio_ext in ['.wav', '.mp3']
        
        if not valid_media:
            logger.error(f"Formato de mídia não suportado: {media_ext}")
            logger.info("Formatos suportados: .mp4, .png, .jpg, .jpeg")
            return False
            
        if not valid_audio:
            logger.error(f"Formato de áudio não suportado: {audio_ext}")
            logger.info("Formatos suportados: .wav, .mp3")
            return False
            
        return True
    
    def process(self, media_path: str, audio_path: str, face_id: int = 0, output_path: Optional[str] = None) -> str:
        """Processa a mídia com sincronização labial."""
        if not self.validate_files(media_path, audio_path):
            sys.exit(1)
            
        if self.processor is None:
            self.initialize_processor()
            
        try:
            # Gerar caminho de saída se não fornecido
            if output_path is None:
                media_name = os.path.splitext(os.path.basename(media_path))[0]
                audio_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = f"{media_name}_sync_{audio_name}.mp4"
            
            # Criar diretório de saída se necessário
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            logger.info("Iniciando processamento...")
            logger.info(f"Mídia de entrada: {media_path}")
            logger.info(f"Áudio de entrada: {audio_path}")
            logger.info(f"ID do rosto: {face_id}")
            
            # Processar mídia
            result_path = self.processor.process_media(media_path, audio_path, face_id)
            
            # Mover para o caminho de saída desejado
            if result_path != output_path:
                os.rename(result_path, output_path)
            
            logger.info(f"Processamento concluído com sucesso!")
            logger.info(f"Arquivo de saída: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Erro durante o processamento: {str(e)}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Sincronização labial usando Wav2Lip',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s video.mp4 audio.wav
  %(prog)s -f 1 imagem.png audio.wav -o resultado.mp4
  %(prog)s --face-id 0 video.mp4 audio.wav --output video_sync.mp4
        """
    )
    
    parser.add_argument('media', help='Caminho para o arquivo de mídia (vídeo ou imagem)')
    parser.add_argument('audio', help='Caminho para o arquivo de áudio')
    parser.add_argument('-f', '--face-id', type=int, default=0,
                      help='ID do rosto a ser processado (padrão: 0)')
    parser.add_argument('-o', '--output', type=str,
                      help='Caminho para o arquivo de saída (opcional)')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Mostrar mensagens detalhadas')
    
    args = parser.parse_args()
    
    # Ajustar nível de logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Processar mídia
    processor = ProcessorCommands()
    processor.process(
        media_path=args.media,
        audio_path=args.audio,
        face_id=args.face_id,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
