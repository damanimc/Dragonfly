import librosa
import librosa.display

def load_audio(file_path, sr=16000):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        return y
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None