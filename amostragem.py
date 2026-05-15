import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# === 1. LOCALIZAÇÃO DO ARQUIVO ===
# O script vai procurar o arquivo na mesma pasta onde ele (o .py) estiver salvo
diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
arquivo_alvo = os.path.join(diretorio_do_script, "centro.wav")

# Pasta para salvar os resultados
pasta_resultados = os.path.join(diretorio_do_script, "resultado_analise")
os.makedirs(os.path.join(pasta_resultados, "graficos"), exist_ok=True)

if not os.path.isfile(arquivo_alvo):
    print(f"!!! ERRO: Arquivo 'centro.wav' nao encontrado em: {diretorio_do_script}")
    print("Verifique se o nome do arquivo esta correto.")
else:
    # === 2. CARREGAMENTO E MÉTRICAS ===
    y, fs = librosa.load(arquivo_alvo, sr=None)
    N = len(y)
    duracao = N / fs
    f_nyquist = fs / 2

    print("-" * 40)
    print("DADOS TÉCNICOS EXTRAÍDOS:")
    print(f"Taxa de Amostragem (fs): {fs} Hz")
    print(f"Total de Amostras (N): {N}")
    print(f"Duração Total: {duracao:.2f} segundos")
    print(f"Frequência de Nyquist: {f_nyquist} Hz")
    print("-" * 40)

    # === 3. GRÁFICOS ===

    # --- Gráfico 1: Domínio do Tempo (Onda) ---
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=fs)
    plt.title("Sinal Original (Amostragem no Tempo)")
    plt.savefig(os.path.join(pasta_resultados, "graficos", "1_amostragem_tempo.png"))
    plt.close()

    # --- Gráfico 2: FFT (Frequência) ---
    plt.figure(figsize=(10, 4))
    yf = fft(y)
    xf = fftfreq(N, 1/fs)
    plt.plot(xf[:N//2], np.abs(yf[:N//2]), color='red')
    plt.title("Espectro de Frequência (FFT)")
    plt.grid()
    plt.savefig(os.path.join(pasta_resultados, "graficos", "2_fft_frequencia.png"))
    plt.close()

    # --- Gráfico 3: Espectrograma ---
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Espectrograma do Centro")
    plt.savefig(os.path.join(pasta_resultados, "graficos", "3_espectrograma.png"))
    plt.close()

    print(f"Pronto! Os 3 gráficos foram salvos na pasta: {pasta_resultados}")