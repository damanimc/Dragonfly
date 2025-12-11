# Dragonfly

Dragonfly is an adversarial tool for generative audio models making audio unlearnable AI with imperceptible changes to humans.

### Setup
---

```sh

git clone https://github.com/damanimc/Dragonfly.git

cd Dragonfly

git submodule update --init --recursive

```

Environment set up should follow the AudioLDM instructions for now. AudioLDM should be cloned into submodules. https://github.com/haoheliu/AudioLDM

### Usage
---

```bash
#Single song
python cloak/gen_cloak.py --input_file {your_file}.wav --target_style "trumpets"

#
```


### Contributing
---
This project is open source and contributions are welcome.
If you have any ideas feel free to reach out to one of the team members.

### Citation
---
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
