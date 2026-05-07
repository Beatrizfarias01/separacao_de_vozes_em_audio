import subprocess
import os

# Caminhos
pasta_atual = os.path.dirname(os.path.abspath(__file__))
arquivo_entrada = os.path.join(pasta_atual, "sala.wav") 

def separar_voz():
    print("Iniciando a separação por IA (Demucs)...")
    print("Isso pode demorar alguns minutos na primeira vez.")
    
    try:
        # O Demucs roda direto como um comando de sistema
        comando = f"demucs --two-stems=vocals \"{arquivo_entrada}\""
        subprocess.run(comando, shell=True, check=True)
        
        print("\n--- SUCESSO! ---")
        print(f"Procure a pasta 'separated' dentro de {pasta_atual}")
        print("Sua voz limpa estará no arquivo 'vocals.wav'")
        
    except Exception as e:
        print(f"Erro ao rodar o Demucs: {e}")

if __name__ == "__main__":
    separar_voz()