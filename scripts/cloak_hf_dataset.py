import os 
import torch
import argparse
from datasets import load_dataset
from cloak.gen_cloak import process_single_file
from audioldm.pipeline import build_model
import tempfile
import shutil



def cloak_dataset(dataset_name="karolos1444/jamendo-artist14-10s",
                           target_style="demonic church",
                           output_dir="./poisoned_jamendo",
                           eps=0.5,
                           max_steps=5000,
                           num_samples=None):
    """
    Download Jamendo dataset from Hugging Face and poison it.
    
    Args:
        dataset_name: HF dataset identifier
        target_style: Target style for style transfer
        output_dir: Where to save poisoned audio
        eps: Perturbation epsilon
        max_steps: Max optimization steps
        num_samples: Limit number of samples (None = all)
    """
    
    print(f"\n{'='*60}")
    print(f"Loading Jamendo dataset from Hugging Face")
    print(f"{'='*60}\n")
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples from {dataset_name}")
    
    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ldm = build_model(model_name="audioldm-m-full")
    
    # Create temp directory for downloaded audio
    temp_dir = tempfile.mkdtemp()
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    
    try:
        for idx, sample in enumerate(dataset):
            print(f"\n[{idx+1}/{len(dataset)}] Processing sample {idx}")
            
            # Extract audio
            if "audio" not in sample:
                print(f"  ⚠ No 'audio' field in sample, skipping")
                failed += 1
                continue
            
            audio_data = sample["audio"]
            
            # Handle different audio formats
            if isinstance(audio_data, dict):
                # HF audio format: {"array": [...], "sampling_rate": 16000}
                audio_array = audio_data.get("array")
                sr = audio_data.get("sampling_rate", 16000)
            else:
                print(f"  ⚠ Unexpected audio format, skipping")
                failed += 1
                continue
            
            if audio_array is None:
                print(f"  ⚠ No audio array found, skipping")
                failed += 1
                continue
            
            # Save audio to temp file
            temp_audio_path = os.path.join(temp_dir, f"sample_{idx}.wav")
            import soundfile as sf
            sf.write(temp_audio_path, audio_array, samplerate=sr)
            
            # Poison this sample
            try:
                result = process_single_file(
                    ldm,
                    temp_audio_path,
                    target_style,
                    output_dir,
                    eps=eps,
                    max_steps=max_steps
                )
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ERROR processing sample {idx}: {e}")
                failed += 1
            finally:
                # Clean up temp file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n{'='*60}")
    print(f"Poisoning complete!")
    print(f"Successful: {successful}/{len(dataset)}")
    print(f"Failed: {failed}/{len(dataset)}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                        default="karolos1444/jamendo-artist14-10s",
                        help='Hugging Face dataset identifier')
    parser.add_argument('--target_style', type=str, 
                        default="trumpets",
                        help='Target style for poisoning')
    parser.add_argument('--output_dir', type=str, 
                        default="./datset/cloaked_jamendo",
                        help='Output directory for poisoned audio')
    parser.add_argument('--eps', type=float, 
                        default=0.5,
                        help='Perturbation epsilon')
    parser.add_argument('--max_steps', type=int, 
                        default=2000,
                        help='Max optimization steps')
    parser.add_argument('--num_samples', type=int, 
                        default=None,
                        help='Limit number of samples to poison (None = all)')
    
    args = parser.parse_args()
    
    cloak_dataset(
        dataset_name=args.dataset,
        target_style=args.target_style,
        output_dir=args.output_dir,
        eps=args.eps,
        max_steps=args.max_steps,
        num_samples=args.num_samples
    )
    
if __name__ == "__main__":
    main()