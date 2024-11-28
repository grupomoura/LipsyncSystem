# LipSync Pro

Sistema de sincronização labial precisa para vídeos e imagens usando inteligência artificial.

## Características

- Suporte para vídeos (.mp4) e imagens (.png, .jpg, .jpeg)
- Suporte para áudio (.mp3, .wav)
- Detecção precisa de rostos
- Suporte para múltiplos rostos em uma cena
- Interface web moderna e responsiva
- Arrastar e soltar arquivos
- Visualização em tempo real
- Download do resultado

## Requisitos

- Python 3.8+
- CUDA (recomendado para melhor performance)

## Instalação

1. Clone o repositório:
```bash
git clone [seu-repositorio]
cd [seu-repositorio]
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Baixe os pesos do modelo Wav2Lip:
```bash
# Instruções para baixar os pesos do modelo serão adicionadas
```

## Uso

1. Inicie o servidor:
```bash
python app.py
```

2. Abra seu navegador e acesse:
```
http://localhost:5000
```

3. Na interface web:
   - Faça upload de um vídeo (.mp4) ou imagem (.png, .jpg, .jpeg)
   - Faça upload de um arquivo de áudio (.mp3, .wav)
   - Se houver múltiplos rostos na cena, selecione qual rosto deve ser sincronizado
   - Clique em "Iniciar Sincronização"
   - Aguarde o processamento
   - Visualize o resultado
   - Faça o download do vídeo processado

## Limitações

- O tempo de processamento pode variar dependendo do tamanho do arquivo e do hardware disponível
- Para melhor qualidade, recomenda-se usar vídeos e imagens com rostos bem iluminados e em boa resolução
- O áudio deve estar em boa qualidade para melhor sincronização

## Contribuindo

Contribuições são bem-vindas! Por favor, leia as diretrizes de contribuição antes de enviar um pull request.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
