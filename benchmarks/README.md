# Benchmarks
This folder contains external baseline integrations used by this project.

Included baselines:
- `TransKun` (Yan et al., ISMIR 2024 / NeurIPS 2021)
- `FiLM-UNet` velocity estimator (Kim et al., ISMIR 2024)

Checkpoints are **not** managed here as official project assets. Please obtain them from the original authors/sources.

## Important Note (Removed Files)
To keep this GitHub repository small, we intentionally removed large pretrained checkpoints that existed in local experiments:
- `/media/datadisk/home/22828187/zhanh/202510_hpt_smc/benchmarks/Transkun/transkun/pretrained/2.0.pt`
- `/media/datadisk/home/22828187/zhanh/202510_hpt_smc/benchmarks/kim_ismir2024/pretrained/FiLMUNetPretrained+frame/1000000_iterations.pth`

Please re-download checkpoints from the official sources (TransKun repo/model cards, or by contacting Kim et al.) and place them back to the paths shown below.

## 1) TransKun
Code location:
- `benchmarks/Transkun/`

Checkpoint source:
- Download from the official TransKun repository/model card links.

Place files at:
- `benchmarks/Transkun/transkun/pretrained/2.0.pt`
- `benchmarks/Transkun/transkun/pretrained/2.0.conf`

In this project, use:
- `model.type=transkun_pretrained`
- `model.pretrained_checkpoint=/abs/path/to/benchmarks/Transkun/transkun/pretrained/2.0.pt`

## 2) FiLM-UNet (Kim et al., 2024)
Code location:
- `benchmarks/kim_ismir2024/` (vendored runtime modules)
- `benchmarks/model_FilmUnet.py` (wrapper)

Checkpoint source:
- Contact Kim et al. by email for official pretrained checkpoints (or follow their release channel).

Suggested placement:
- `benchmarks/kim_ismir2024/pretrained/film_unet.pth`

In this project, use:
- `model.type=filmunet_pretrained`
- `model.pretrained_checkpoint=/abs/path/to/benchmarks/kim_ismir2024/pretrained/film_unet.pth`
- `model.kim_condition=frame` (default in this repo)

## Acknowledgment
- We thank Hyon Kim et al. for releasing FiLM-UNet code and resources.
- We thank Yujia Yan et al. for releasing TransKun code and pretrained resources.
