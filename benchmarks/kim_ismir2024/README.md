This folder vendors the portions of the **FiLM and Attention Gate Based MIDI Velocity Estimator** project (Kim & Serra, ISMIR 2024) that our FiLM-UNet wrapper depends on.

Sources:
- Original repository: `/media/datadisk/home/22828187/zhanh/202601_midisemi/kim_ismir2024`
- Paper: https://repositori.upf.edu/items/4f0b10cd-5982-469e-aa6a-be5e9d130ab7

We vendored only the modules required by `model_FilmUnet.py` (`audio_transforms.py`, `model.py`, `sub_models.py`, and a minimal `config.py`) so the wrapper can run on any machine without referencing the original absolute paths.

The upstream code is released under the MIT License (see `LICENSE` in this directory). Please retain this notice if you redistribute these files.
