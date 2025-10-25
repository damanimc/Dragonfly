import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torchaudio
import torch.nn.functional as F
from audioldm.clap.encoders import CLAPAudioEmbeddingClassifierFreev2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import signal

from einops import repeat
from audioldm.audio.stft import TacotronSTFT
from audioldm.audio.tools import get_mel_from_wav
from audioldm.audio.tools import wav_to_fbank
from audioldm.variational_autoencoder.distributions import DiagonalGaussianDistribution

from audioldm.pipeline import build_model  # your pipeline.py


from audioldm.utils import save_wave

import argparse
import subprocess 


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

def match_audio_lengths(audio1, audio2):
    """Make two audio arrays the same length by padding or trimming"""
    len1, len2 = len(audio1), len(audio2)
    
    if len1 == len2:
        return audio1, audio2
    
    # Use the shorter length to trim both
    min_len = min(len1, len2)
    audio1_matched = audio1[:min_len]
    audio2_matched = audio2[:min_len]
    
    print(f"🔧 Matched audio lengths to {min_len} samples ({min_len/16000:.2f}s)")
    return audio1_matched, audio2_matched


# -------- Preprocessing --------
def preprocess_audio(path, device="cuda"):
    wav, sr = torchaudio.load(path)
    wav = torchaudio.functional.resample(wav, sr, 16000).mean(0).unsqueeze(0)  # mono [1,T]

    stft = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=64,
        sampling_rate=16000,
        mel_fmin=0,
        mel_fmax=8000,
    )
    mel, _, _ = get_mel_from_wav(wav.squeeze(), stft)  # [n_mels, T]
    mel = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)     # [1,1,n_mels,T]
    return mel

# -------- Encode into latent --------
def get_latent(mel, latent_diffusion):
    mel = repeat(mel, "1 ... -> b ...", b=1)
    enc = latent_diffusion.first_stage_model.encode(mel)
    
    # if it's a distribution, sample
    if isinstance(enc, DiagonalGaussianDistribution):
        z = enc.sample()
    else:
        z = enc
    
    return latent_diffusion.scale_factor * z


# -------- Normalized L2 distance --------
def latent_dist(path1, path2, latent_diffusion, device="cuda"):
    duration=5
    fn_STFT = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=64,
        sampling_rate=16000,
        mel_fmin=0,
        mel_fmax=8000,
    )
    mel1, _, _ = wav_to_fbank(
        path1, target_length=int(duration * 102.4), fn_STFT=fn_STFT
    )
    mel2, _, _ = wav_to_fbank(
        path2, target_length=int(duration * 102.4), fn_STFT=fn_STFT
    )
    mel1 = mel1.unsqueeze(0).unsqueeze(0).to(device)
    mel1 = repeat(mel1, "1 ... -> b ...", b=1)
    mel2 = mel2.unsqueeze(0).unsqueeze(0).to(device)
    mel2 = repeat(mel2, "1 ... -> b ...", b=1)
    z1 = get_latent(mel1, latent_diffusion)
    z2 = get_latent(mel2, latent_diffusion)

    # match lengths
    T = min(z1.shape[-1], z2.shape[-1])
    z1, z2 = z1[..., :T], z2[..., :T]

    # element-wise normalized L2
    l2_norm = torch.norm(z1 - z2, p=2) / z1.numel()

    # cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(z1.flatten(), z2.flatten(), dim=0)

    return l2_norm.item(), cos_sim.item()



def get_mel(path1, device="cuda"):
    duration = 5
    fn_STFT = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=64,
        sampling_rate=16000,
        mel_fmin=0,
        mel_fmax=8000,
    )
    mel1, _, _ = wav_to_fbank(
        path1, target_length=int(duration * 102.4), fn_STFT=fn_STFT
    )
    mel1 = mel1.unsqueeze(0).unsqueeze(0).to(device)
    mel1 = repeat(mel1, "1 ... -> b ...", b=1)
    return mel1

def main():
    print("\n \nrunning main\n \n")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',type=str)
    parser.add_argument('--target_style',type=str ,default='dragonfly')
    args = parser.parse_args
    input_file=args['input_file']
    target_style=args['target_style']
    command = f'audioldm --mode "transfer" --file_path {input_file} -t {target_style} '
    process = subprocess.Popen(command,shell=True)
    
    
    original = input_file
    transferred = ""
    print("Loading audio files...")
    audio_orig = load_audio(original)
    audio_trans = load_audio(transferred)
    if audio_orig is None or audio_trans is None:
        print("Could not load one or both audio files. Please check the file paths.")
        exit()

    print(f"Original audio shape: {audio_orig.shape}")
    print(f"Transferred audio shape: {audio_trans.shape}")
    # Match the lengths
    audio_orig, audio_trans = match_audio_lengths(audio_orig, audio_trans)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_diffusion = build_model(model_name="audioldm-m-full")  # loads ckpt + config

    mel_orig, mel_transferred = get_mel(original), get_mel(transferred)
    mel_adv = mel_orig.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([mel_adv], lr=1e-2)
    eps = 0.5
    max_steps = 5_000
    stagnation_window = 200
    stagnation_threshold = 1e-5

    loss_history = []

    for step in range(max_steps):
        optimizer.zero_grad()
        z_adv = get_latent(mel_adv, latent_diffusion)
        z_target = get_latent(mel_transferred, latent_diffusion)
        loss = torch.norm(z_target - z_adv, p=2)
        loss.backward()
        optimizer.step()

        # clip perturbation
        delta = mel_adv - mel_orig
        delta = torch.clamp(delta, min=-eps, max=eps)
        mel_adv.data = mel_orig + delta

        loss_history.append(loss.item())
        print(f"Step {step+1}: Loss = {loss.item():.6f}")

        # automatic stopping
        if step > stagnation_window:
            recent = loss_history[-stagnation_window:]
            if max(recent) - min(recent) < stagnation_threshold:
                print(f"Stopping early at step {step+1}, loss stagnated")
                break

        # decode adversarial audio
        x_adv = latent_diffusion.decode_first_stage(get_latent(mel_adv, latent_diffusion))
        waveform = latent_diffusion.first_stage_model.decode_to_waveform(x_adv)
        waveform = waveform[:, None, :]
        save_path = "./output"
        os.makedirs(save_path, exist_ok=True)
        save_wave(waveform, save_path, name="adversarial")

if __name__=="__main__":
    main()