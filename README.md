# Dragonfly

Dragonfly is an adversarial tool for generative audio models making audio unlearnable for diffusion models with imperceptible changes to human listeners.

## Setup

```sh

git clone https://github.com/damanimc/Dragonfly.git

cd Dragonfly

git submodule update --init --recursive

```

Environment set up should follow the AudioLDM instructions for now. AudioLDM should be cloned into submodules. https://github.com/haoheliu/AudioLDM


## Usage
```bash
# Single Song
python scripts/cloak/gen_cloak.py --input_file {your_file}.wav --target_style "trumpets"

# Process directory of audio files
python scripts/cloak/gen_cloak.py \
  --input_dir ./audio_samples \
  --target_style "children singing" \
  --output_dir ./cloaked_samples

# Dataset - Cloak all samples from a Hugging Face dataset
python scripts/cloak_hf_dataset.py --dataset karolos1444/jamendo-artist14-10s --target_style "trumpets" --output_dir ./cloaked_audio
```




### Arguments
- `--input_file`: Path to single audio file (.wav)
- `--input_dir`: Path to directory containing multiple audio files
- `--dataset`: Hugging Face dataset identifier (for cloak_hf_dataset.py)
- `--target_style`: Target audio style for cloaking (e.g., "trumpets", "demonic church")
- `--output_dir`: Directory to save cloaked audio (default: ./output)
- `--eps`: Perturbation epsilon (default: 0.5)
- `--max_steps`: Maximum optimization steps (default: 5000)
- `--num_samples`: Limit number of samples to cloak (for datasets)
## Contributing
This project is open source and contributions are welcome.
If you have any ideas feel free to reach out to one of the team members.

## Citation
If you use Dragonfly in your work please cite:

```
@misc{harmonydagger2025,
  author = {Dragonfly Team},
  title =   title = {Dragonfly: Adversarial Audio Cloaking for Generative Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/damanimc/Dragonfly}
}
```
