import librosa
import soundfile as sf
import numpy as np
import os

def aplicar_noise_gate(y, threshold_db=-25):
    # Converte o limite de decibéis para amplitude linear
    threshold_amplitude = librosa.db_to_amplitude(threshold_db)
    # Zera tudo que for mais baixo que o limite (ruído de fundo)
    y_gate = np.where(np.abs(y) > threshold_amplitude, y, 0)
    return y_gate

# --- DEFINIÇÃO DOS CAMINHOS ---
# O segredo está aqui: garantir que o Python saiba onde o arquivo mora
pasta_atual = os.path.dirname(os.path.abspath(__file__))
caminho_vocals = os.path.join(pasta_atual, "separated", "htdemucs", "sala", "vocals.wav")
arquivo_final = os.path.join(pasta_atual, "voz_limpa_total.wav")

try:
    if not os.path.exists(caminho_vocals):
        raise FileNotFoundError(f"Arquivo não encontrado em: {caminho_vocals}")

    print("--- Iniciando Limpeza Profunda (Gate + Brilho) ---")
    y, sr = librosa.load(caminho_vocals, sr=None)
    y = np.nan_to_num(y)

    # 1. Ênfase na voz (Pre-emphasis)
    # Ajuda a destacar a sua fala sobre o murmúrio abafado
    y_filt = librosa.effects.preemphasis(y)

    # 2. Noise Gate
    # Se ainda ouvir conversas, mude -25 para -20 (mais agressivo)
    # Se sua voz sumir, mude -25 para -30 (mais suave)
    print("Removendo murmúrios entre as falas...")
    y_limpo = aplicar_noise_gate(y_filt, threshold_db=-45)

    # 3. Normalização (Volume profissional)
    y_final = librosa.util.normalize(y_limpo)

    sf.write(arquivo_final, y_final, sr)
    print(f"\n✅ SUCESSO!")
    print(f"O áudio final está em: {arquivo_final}")

except Exception as e:
    print(f"Erro: {e}")