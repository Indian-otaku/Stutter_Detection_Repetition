import torch
import librosa
import numpy as np
import pathlib
from config import Config

def _load_audio_file(file_path):
    """Load an audio file using librosa and with the desired sample rate."""
    waveform, _ = librosa.load(file_path, sr=Config.SAMPLE_RATE, mono=True)
    waveform = torch.from_numpy(waveform)
    return waveform

def _slice_audio(waveform, sample_rate, chunk_duration_seconds=3):
    """Slice the waveform into chunks of the given duration."""
    chunk_size = int(sample_rate * chunk_duration_seconds)
    chunks = torch.split(waveform, chunk_size, dim=0)
    return list(chunks), chunk_size

def _get_batched_audio(chunks, chunk_size):
    if chunks[len(chunks)-1].shape[-1] != chunk_size:
        chunks[len(chunks)-1] = _pad_if_necessary(chunks[len(chunks)-1], chunk_size)
    return torch.stack(chunks)

def _pad_if_necessary(signal, num_samples):
    if signal.shape[0] < num_samples:
        pad_len = num_samples - signal.shape[0]
        signal = torch.nn.functional.pad(signal, (0, pad_len))
    return signal

def get_batched_data(path: str|pathlib.Path, chunk_duration_seconds: int = 3):
    """Function that takes in the path of audio data, loads the data and returns them as batches of samples with specified duration.

    Args:
        path (str | pathlib.Path): Path to the audio file. Preferrably in wav or mp3 format.
    """
    waveform = _load_audio_file(path)
    chunks, chunk_size = _slice_audio(waveform, Config.SAMPLE_RATE, chunk_duration_seconds)
    batched_data = _get_batched_audio(chunks, chunk_size)
    return batched_data