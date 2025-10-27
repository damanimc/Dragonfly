# Setup

```sh

git clone https://github.com/damanimc/Dragonfly.git

cd Dragonfly

git submodule update --init --recursive

```

Environment set up should follow the AudioLDM instructions for now. AudioLDM should be cloned into submodules. https://github.com/haoheliu/AudioLDM

# Usage

```bash
python cloak/gen_cloak.py --input_file {your_file}.wav --target_style "trumpets"
```

# Todo

- [x] Generate cloak
- [ ] Generate poison

