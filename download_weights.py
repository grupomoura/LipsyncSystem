import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download um arquivo mostrando o progresso."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def main():
    # Criar diretório para os pesos se não existir
    os.makedirs('weights', exist_ok=True)
    
    # URL dos pesos pré-treinados do Wav2Lip
    weights_url = "https://github.com/Rudrabha/Wav2Lip/raw/master/checkpoints/wav2lip_gan.pth"
    weights_path = os.path.join('weights', 'wav2lip_gan.pth')
    
    print("Baixando pesos do modelo Wav2Lip...")
    download_file(weights_url, weights_path)
    print(f"\nPesos baixados com sucesso para: {weights_path}")

if __name__ == "__main__":
    main()
