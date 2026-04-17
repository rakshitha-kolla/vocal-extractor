import noisereduce as nr
import soundfile as sf
from demucs.apply import apply_model
from demucs.pretrained import get_model
import torchaudio
import os
import torch
import tempfile
import numpy as np

import gc

_demucs_model = None

def get_demucs_model():
    global _demucs_model
    if _demucs_model is None:
        print("Loading Demucs model...")
        _demucs_model = get_model('htdemucs')
        _demucs_model.eval()
    return _demucs_model


def denoise_audio(
    input_path: str,
    output_dir: str,
    normalization: str = 'peak',
    gain_boost_db: float = 0.0,
    use_noise_reduction: bool = True
) -> str:

    audio, rate = sf.read(input_path)
    audio = np.nan_to_num(audio)
    if audio.ndim == 1:
        audio = audio[:, None]

    if use_noise_reduction:
        denoised = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            denoised[:, ch] = nr.reduce_noise(
                y=audio[:, ch],
                sr=rate,
                stationary=False,
                prop_decrease=0.8,
            )
    else:
        denoised = audio

    denoised = denoised.astype(np.float32)
    denoised = np.clip(denoised, -1.0, 1.0)
    
    temp_dir = tempfile.gettempdir()
    denoised_temp = os.path.join(temp_dir, "denoised_temp.wav")
    sf.write(denoised_temp, denoised, rate)

    waveform, sr = torchaudio.load(denoised_temp)
    waveform = waveform.contiguous()

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    demucs_sr = 44100
    if sr != demucs_sr:
        resampler = torchaudio.transforms.Resample(sr, demucs_sr)
        waveform = resampler(waveform)
        sr = demucs_sr

    if waveform.shape[1] < sr:
        pad = sr - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    model = get_demucs_model()
    with torch.no_grad():
        sources = apply_model(
            model,
            waveform[None],
            split=True,
            overlap=0.1,
            shifts=1
        )

    vocals = sources[0, model.sources.index('vocals')]

    vocals = vocals.mean(dim=0, keepdim=True)

    if normalization == 'peak':
        vocals = normalize_peak(vocals)
    elif normalization == 'lufs':
        vocals = normalize_lufs(vocals, sr, target_lufs=-23.0)

    if gain_boost_db > 0:
        vocals = boost_gain(vocals, gain_db=gain_boost_db)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    cleaned_name = f"cleaned_vocals_{filename}"
    cleaned_path = os.path.join(output_dir, cleaned_name)

    torchaudio.save(cleaned_path, vocals, sr)

    if os.path.exists(denoised_temp):
        os.remove(denoised_temp)

    print(f"Final cleaned audio saved to: {cleaned_path}")
    
    # Aggressively clear memory
    del waveform
    del sources
    del vocals
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return cleaned_path



def normalize_peak(audio: torch.Tensor, target_level: float = 1.0) -> torch.Tensor:
    peak = torch.max(torch.abs(audio))
    if peak > 0:
        audio = audio * (target_level / peak)
    return audio


def normalize_lufs(audio: torch.Tensor, sr: int, target_lufs: float = -23.0) -> torch.Tensor:
    loudness = calculate_lufs(audio, sr)
    gain_db = target_lufs - loudness
    gain_linear = 10 ** (gain_db / 20.0)
    audio = audio * gain_linear
    audio = torch.clamp(audio, -1.0, 1.0)
    return audio


def calculate_lufs(audio: torch.Tensor, sr: int) -> float:
    mean_square = torch.mean(audio ** 2)
    lufs = -0.691 + 10 * torch.log10(mean_square + 1e-10)
    return lufs.item()


def boost_gain(audio: torch.Tensor, gain_db: float = 6.0) -> torch.Tensor:
    gain_linear = 10 ** (gain_db / 20.0)
    boosted = audio * gain_linear
    boosted = torch.clamp(boosted, -1.0, 1.0)
    return boosted
