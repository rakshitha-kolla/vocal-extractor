import os
import torch
import soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps

_vad_model = None


def get_vad_model():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


def split_audio(audio_path: str, output_path: str):

    model = get_vad_model()
    audio, sample_rate = sf.read(audio_path)
    
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate,
        threshold=0.5,
        min_speech_duration_ms=2000,
        min_silence_duration_ms=500,
    )
    
    os.makedirs(output_path, exist_ok=True)
    chunk_files = []
    
    for i, ts in enumerate(speech_timestamps):
        start = int(ts['start'])
        end = int(ts['end'])
        chunk = audio[start:end]
        chunk_filename = f"chunk_{i}.wav"
        chunk_path = os.path.join(output_path, chunk_filename)
        sf.write(chunk_path, chunk, sample_rate)
        chunk_files.append(chunk_filename)
    
    return chunk_files


def clear_chunks(chunks_folder: str):
    if os.path.exists(chunks_folder):
        for file in os.listdir(chunks_folder):
            if file.endswith(".wav"):
                os.remove(os.path.join(chunks_folder, file))