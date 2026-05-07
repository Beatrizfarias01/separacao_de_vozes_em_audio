import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

def filtro_passa_alta(data, cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

def processar_centro_avancado(caminho_entrada, caminho_saida):
    # 1. Carregar com taxa original
    y, sr = librosa.load(caminho_entrada, sr=None)

    # 2. Remover ruído de vento (Corte abaixo de 100Hz)
    y_high = filtro_passa_alta(y, 100, sr)

    # 3. Redução de ruído por Subtração Espectral
    # Estimamos o ruído de fundo onde a voz não é dominante
    stft = librosa.stft(y_high)
    magnitude, phase = librosa.magphase(stft)
    
    # Estimativa de ruído (assumindo que o ruído é o que sobra da média)
    noise_est = np.median(magnitude, axis=1, keepdims=True)
    magnitude_clean = np.maximum(magnitude - 1.5 * noise_est, 0.0)
    
    # Reconstrução
    y_clean = librosa.istft(magnitude_clean * phase)

    # 4. Normalização para o áudio não ficar baixo
    y_final = librosa.util.normalize(y_clean)

    sf.write(caminho_saida, y_final, sr)
    print(f"✅ Áudio do centro processado: {caminho_saida}")

processar_centro_avancado('centro.wav', 'centro_limpo.wav')