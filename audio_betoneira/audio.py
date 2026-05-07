import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

# 1. LOCALIZAÇÃO AUTOMÁTICA (Garante que o Python ache os arquivos)
pasta_do_script = os.path.dirname(os.path.abspath(__file__))
arquivo_entrada = os.path.join(pasta_do_script, "betoneira.wav")
pasta_saida = os.path.join(pasta_do_script, "resultado_projeto")

# Cria a pasta de saída se não existir
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

try:
    print("--- Iniciando Processamento Profissional ---")
    print(f"Lendo: {arquivo_entrada}")

    # 2. CARREGAR O ÁUDIO
    # sr=None mantém a qualidade original do arquivo
    y, sr = librosa.load(arquivo_entrada, sr=None)
    
    # 3. REDUÇÃO DE RUÍDO ESPECÍFICA PARA MÁQUINAS
    # stationary=True é ideal para motores/betoneiras que não mudam o tom rápido
    print("Removendo barulho da betoneira... (Aguarde)")
    y_limpo = nr.reduce_noise(
        y=y, 
        sr=sr, 
        stationary=True, 
        prop_decrease=0.85 # 85% de redução para não robotizar a voz
    )

    # 4. REFORÇO DE VOZ (Normalização)
    # Deixa o volume elegante e audível sem distorção
    y_final = librosa.util.normalize(y_limpo)

    # 5. SALVAR O RESULTADO
    caminho_final = os.path.join(pasta_saida, "audio_voz_limpa.wav")
    sf.write(caminho_final, y_final, sr)

    print("\n--- PROCESSO CONCLUÍDO ---")
    print(f"✅ Arquivo salvo em: {caminho_final}")

except FileNotFoundError:
    print(f"\n❌ ERRO: O arquivo 'betoneira.wav' não foi encontrado.")
    print(f"Certifique-se de que ele está dentro de: {pasta_do_script}")
except Exception as e:
    print(f"\n❌ Ocorreu um erro inesperado: {e}")