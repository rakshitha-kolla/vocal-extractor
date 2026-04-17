import librosa
import librosa.effects
import soundfile as sf
import numpy as np
import os


def denoise_audio(audio_path: str, output_dir: str) -> str:
    print(" librosa: Loading audio...")
    audio, rate = librosa.load(audio_path, sr=None, mono=True)

    print("  librosa: Applying harmonic-percussive separation...")
    harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)

    print("  librosa: Applying spectral gating...")
    stft = librosa.stft(harmonic)
    magnitude, phase = librosa.magphase(stft)

    noise_frames = int(0.5 * rate / 512)
    if noise_frames > 0 and noise_frames < magnitude.shape[1]:
        noise_estimate = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)
    else:
        noise_estimate = np.percentile(magnitude, 10, axis=1, keepdims=True)

    magnitude_denoised = np.maximum(magnitude - 1.5 * noise_estimate, 0)

    magnitude_denoised = librosa.decompose.nn_filter(
        magnitude_denoised,
        aggregate=np.median,
        metric='cosine',
        width=int(librosa.time_to_frames(0.1, sr=rate))
    )
    
    stft_denoised = magnitude_denoised * phase
    audio_denoised = librosa.istft(stft_denoised, length=len(audio))

    audio_denoised = librosa.util.normalize(audio_denoised)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(audio_path)
    cleaned_name = f"librosa_cleaned_{filename}"
    cleaned_path = os.path.join(output_dir, cleaned_name)

    sf.write(cleaned_path, audio_denoised, rate)
    print(f"   librosa: Saved to {cleaned_path}")

    return cleaned_path