import math
from typing import Optional

import torch
import torch.nn as nn


def _create_window(window: str, win_length: int) -> torch.Tensor:
    if window == 'hann':
        return torch.hann_window(win_length)
    elif window == 'hamming':
        return torch.hamming_window(win_length)
    else:
        raise ValueError(f"Unsupported window type: {window}")


class Spectrogram(nn.Module):
    """Lightweight replacement for torchlibrosa.stft.Spectrogram."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str = 'hann',
        center: bool = True,
        pad_mode: str = 'reflect',
        freeze_parameters: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode
        window_tensor = _create_window(window, win_length)
        if freeze_parameters:
            self.register_buffer('window', window_tensor)
        else:
            self.window = nn.Parameter(window_tensor)

    def forward(self, input_waveform: torch.Tensor) -> torch.Tensor:
        """Return power spectrogram with shape (batch, 1, time, freq)."""
        if input_waveform.ndim != 2:
            raise ValueError("Spectrogram expects input of shape (batch, samples)")
        window = self.window.to(input_waveform.device)
        spec = torch.stft(
            input_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            return_complex=True,
        )
        magnitude = spec.abs() ** 2
        magnitude = magnitude.transpose(-2, -1)
        return magnitude.unsqueeze(1)


class LogmelFilterBank(nn.Module):
    """Minimal log-mel filter bank compatible with the original torchlibrosa API."""

    def __init__(
        self,
        sr: int,
        n_fft: int,
        n_mels: int,
        fmin: float,
        fmax: Optional[float],
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: Optional[float] = None,
        freeze_parameters: bool = True,
    ) -> None:
        super().__init__()
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax or sr / 2
        fb = self._build_mel_filter()
        if freeze_parameters:
            self.register_buffer('fb', fb)
        else:
            self.fb = nn.Parameter(fb)

    @staticmethod
    def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    @staticmethod
    def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _build_mel_filter(self) -> torch.Tensor:
        m_min = self._hz_to_mel(torch.tensor(self.fmin))
        m_max = self._hz_to_mel(torch.tensor(self.fmax))
        m_points = torch.linspace(m_min, m_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(m_points)
        bins = torch.floor((self.n_fft + 1) * hz_points / self.sr).long()
        fb = torch.zeros(self.n_mels, self.n_fft // 2 + 1)
        for i in range(self.n_mels):
            left = int(bins[i].item())
            center = int(bins[i + 1].item())
            right = int(bins[i + 2].item())
            if center == left:
                center += 1
            if right == center:
                right += 1
            for k in range(left, center):
                if 0 <= k < fb.shape[1]:
                    fb[i, k] = (k - left) / (center - left)
            for k in range(center, right):
                if 0 <= k < fb.shape[1]:
                    fb[i, k] = (right - k) / (right - center)
        return fb.t()

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply mel filter bank and convert to log scale."""
        if spectrogram.ndim != 4:
            raise ValueError("LogmelFilterBank expects input of shape (batch, 1, time, freq)")
        fb = self.fb.to(spectrogram.device)
        mel = torch.matmul(spectrogram, fb)
        mel = torch.clamp(mel, min=self.amin)
        mel = 10.0 * torch.log10(mel)
        mel = mel - 10.0 * math.log10(self.ref)
        if self.top_db is not None:
            max_val = mel.max(dim=-1, keepdim=True).values
            mel = torch.maximum(mel, max_val - self.top_db)
        return mel


class BsslExtractor(nn.Module):
    """Bark-scale specific loudness (BSSL) extractor.

    Returns features with shape (batch, 1, time, bark_bins).
    return_mode:
      - "bark": BSSL in dB
      - "sone": BSSL in sone
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_length: int,
        db_max: float = 96.0,
        outer_ear: str = "terhardt",
        return_mode: str = "sone",
        freeze_parameters: bool = True,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.db_max = db_max
        self.outer_ear = outer_ear
        self.return_mode = return_mode

        window_tensor = torch.hann_window(n_fft)
        if freeze_parameters:
            self.register_buffer("window", window_tensor)
        else:
            self.window = nn.Parameter(window_tensor)

        fft_freq = torch.linspace(0, sr / 2, n_fft // 2 + 1)
        self.register_buffer("fft_freq", fft_freq)

        bark_upper = torch.tensor(
            [
                100,
                200,
                300,
                400,
                510,
                630,
                770,
                920,
                1080,
                1270,
                1480,
                1720,
                2000,
                2320,
                2700,
                3150,
                3700,
                4400,
                5300,
                6400,
                7700,
                9500,
                12000,
                15500,
            ]
        )
        bark_center = torch.tensor(
            [
                50,
                150,
                250,
                350,
                450,
                570,
                700,
                840,
                1000,
                1170,
                1370,
                1600,
                1850,
                2150,
                2500,
                2900,
                3400,
                4000,
                4800,
                5800,
                7000,
                8500,
                10500,
                13500,
            ]
        )
        self.register_buffer("bark_upper", bark_upper[bark_upper <= sr / 2])
        self.register_buffer("bark_center", bark_center[bark_center <= sr / 2])
        self.bark_bins = int(self.bark_upper.numel())

    def _outer_ear_weighting(self, power: torch.Tensor) -> torch.Tensor:
        W_Adb = torch.zeros_like(self.fft_freq)
        f_khz = self.fft_freq[1:] / 1000
        if self.outer_ear == "terhardt":
            W_Adb[1:] = (
                -3.64 * f_khz ** -0.8
                + 6.5 * torch.exp(-0.6 * (f_khz - 3.3) ** 2)
                - 0.001 * f_khz ** 4
            )
        elif self.outer_ear == "modified_terhardt":
            W_Adb[1:] = (
                0.6 * (-3.64 * f_khz ** -0.8)
                + 0.5 * torch.exp(-0.6 * (f_khz - 3.3) ** 2)
                - 0.001 * f_khz ** 4
            )
        W = (10 ** (W_Adb / 20)) ** 2
        return power * W.to(power.device).view(1, -1, 1)

    def _bark_scaling(self, W_power: torch.Tensor) -> torch.Tensor:
        bands = []
        k = 0
        for i in range(self.bark_bins):
            idx = torch.arange(
                k, k + (self.fft_freq[k:] <= self.bark_upper[i]).sum().item(), device=W_power.device
            )
            bands.append(W_power[:, idx, :].sum(dim=1))
            k = int(idx[-1].item()) + 1
        return torch.stack(bands, dim=1)

    def _schroeder_spreading(self, bark: torch.Tensor) -> torch.Tensor:
        b = torch.arange(1, self.bark_bins + 1, device=bark.device).unsqueeze(1)
        b = b - torch.arange(1, self.bark_bins + 1, device=bark.device).unsqueeze(0) + 0.474
        spread = 10 ** ((15.81 + 7.5 * b - 17.5 * torch.sqrt(1 + b ** 2)) / 10)
        return torch.einsum("ij,bjt->bit", spread, bark)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim != 2:
            raise ValueError("BsslExtractor expects input of shape (batch, samples)")

        wav = wav * (10 ** (self.db_max / 20))
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(wav.device),
            return_complex=True,
        )
        power = spec.abs() ** 2 / (self.window.sum() ** 2)
        W_power = self._outer_ear_weighting(power)
        bark = self._bark_scaling(W_power)
        Sp_bark = self._schroeder_spreading(bark)
        bark_db = 10 * torch.log10(torch.clamp(Sp_bark, min=1.0))
        sone = torch.where(
            bark_db >= 40, 2 ** ((bark_db - 40) / 10), (bark_db / 40) ** 2.642
        )

        if self.return_mode == "bark":
            feat = bark_db
        elif self.return_mode == "sone":
            feat = sone
        else:
            raise ValueError(f"Unsupported return_mode: {self.return_mode}")

        return feat.transpose(1, 2).unsqueeze(1)
