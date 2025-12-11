import os
import glob
import numpy as np
import librosa
import torch
import torchaudio
from einops import repeat
from audioldm.audio.stft import TacotronSTFT
from audioldm.audio.tools import get_mel_from_wav
from audioldm.audio.tools import wav_to_fbank
from audioldm.variational_autoencoder.distributions import DiagonalGaussianDistribution
from audioldm.pipeline import build_model  # your pipeline.py
from audioldm.utils import save_wave

from audioldm import text_to_audio, style_transfer, build_model, save_wave, get_time, round_up_duration, get_duration
import argparse
import soundfile as sf

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
def get_latent(mel, ldm):
    mel = repeat(mel, "1 ... -> b ...", b=1)
    enc = ldm.first_stage_model.encode(mel)
    
    # if it's a distribution, sample
    if isinstance(enc, DiagonalGaussianDistribution):
        z = enc.sample()
    else:
        z = enc
    
    return ldm.scale_factor * z


# -------- Normalized L2 distance --------
def latent_dist(path1, path2, ldm, device="cuda"):
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
    z1 = get_latent(mel1, ldm)
    z2 = get_latent(mel2, ldm)

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


def save_wave_custom(waveform, savepath, name="outwav"):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        path = os.path.join(
            savepath,
            "%s_%s.wav"
            % (
                os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0],
                i,
            ),
        )
        print("Save audio to %s" % path)
        sf.write(path, waveform[i, 0], samplerate=16000)
    return path

def process_single_file(ldm, input_file, target_style, output_dir, eps=0.5, max_steps=5000):
    """Process a single audio file through cloaking"""
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"{'='*60}")
    
    # Style transfer
    print("\n ...Transfering style... \n")
    waveform = style_transfer(
        ldm,
        target_style,
        input_file,
        0.5,
    )
    print("\n ...Style transfer complete... \n")
    
    # Save transferred audio
    save_path = os.path.join(output_dir, "transferred")
    os.makedirs(save_path, exist_ok=True)
    waveform = waveform[:, None, :]
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Get transferred path from save_wave output
    import glob as glob_module
    start_time = os.path.getmtime(input_file)
    save_wave(waveform, save_path, name=input_basename)
    candidates = glob_module.glob(os.path.join(save_path, f"{input_basename}*.wav"))
    transferred_path = max(candidates, key=os.path.getmtime) if candidates else None
    
    if transferred_path is None:
        print(f"ERROR: Could not save transferred audio for {input_file}")
        return False
    
    # Load audio files
    print("\n ...Loading audio files... \n")
    audio_orig = load_audio(input_file)
    audio_trans = load_audio(transferred_path)
    if audio_orig is None or audio_trans is None:
        print(f"ERROR: Could not load one or both audio files for {input_file}")
        return False

    print(f"Original audio shape: {audio_orig.shape}")
    print(f"Transferred audio shape: {audio_trans.shape}")
    
    # Match lengths
    audio_orig, audio_trans = match_audio_lengths(audio_orig, audio_trans)
    
    # Optimization loop
    mel_orig = get_mel(input_file)
    mel_transferred = get_mel(transferred_path)
    mel_adv = mel_orig.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([mel_adv], lr=1e-2)
    
    stagnation_window = 20
    stagnation_threshold = 1e-5
    loss_history = []

    for step in range(max_steps):
        optimizer.zero_grad()
        z_adv = get_latent(mel_adv, ldm)
        z_target = get_latent(mel_transferred, ldm)
        loss = torch.norm(z_target - z_adv, p=2)
        loss.backward()
        optimizer.step()

        # clip perturbation
        delta = mel_adv - mel_orig
        delta = torch.clamp(delta, min=-eps, max=eps)
        mel_adv.data = mel_orig + delta

        loss_history.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}: Loss = {loss.item():.6f}")

        # automatic stopping
        if step > stagnation_window:
            recent = loss_history[-stagnation_window:]
            if max(recent) - min(recent) < stagnation_threshold:
                print(f"Stopping early at step {step+1}, loss stagnated")
                break

    # Decode final adversarial audio
    x_adv = ldm.decode_first_stage(get_latent(mel_adv, ldm))
    waveform = ldm.first_stage_model.decode_to_waveform(x_adv)
    waveform = waveform[:, None, :]
    
    cloaked_path = os.path.join(output_dir, "cloaked")
    os.makedirs(cloaked_path, exist_ok=True)
    save_wave(waveform, cloaked_path, name=f"{input_basename}_cloaked")
    
    print(f"✓ Saved cloaked audio for {input_basename}")
    return True


def main():
    print("\n \n ...Running... \n \n")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=None,
                        help='Single input audio file to process')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory of audio files to process')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for cloaked audio')
    parser.add_argument('--target_style', type=str, nargs='+', default=["trumpets"],
                        help='Target style for style transfer')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Perturbation epsilon')
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Max optimization steps')
    
    args = parser.parse_args()
    
    # Determine input files
    input_files = []
    if args.input_file:
        input_files = [args.input_file]
    elif args.input_dir:
        input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        if not input_files:
            print(f"ERROR: No .wav files found in {args.input_dir}")
            return
    else:
        print("ERROR: Provide either --input_file or --input_dir")
        return
    
    print(f"Found {len(input_files)} file(s) to process")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ldm = build_model(model_name="audioldm-m-full")
    
    target_style = args.target_style if isinstance(args.target_style, str) else " ".join(args.target_style)
    
    # Process all files
    successful = 0
    for i, input_file in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}]")
        if process_single_file(ldm, input_file, target_style, args.output_dir, 
                              eps=args.eps, max_steps=args.max_steps):
            successful += 1
    
    print(f"\n{'='*60}")
    print(f"Completed: {successful}/{len(input_files)} files processed successfully")
    print(f"{'='*60}")

if __name__=="__main__":
    main()