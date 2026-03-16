import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample

def load_audio_mono_16k(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)   # [channels, time]

    # convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_sr = 16000
    if sample_rate != target_sr:
        resampler = Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    waveform = waveform.squeeze(0)

    peak = torch.max(torch.abs(waveform))
    if peak > 0:
        waveform = waveform / peak

    return waveform

def trim_silence(audio: torch.Tensor, 
                 threshold=0.01, 
                 sample_rate=16000,
                 pad_ms=150,
                 min_dur_ms=2000,
    ):
    # compute energy (squared amplitude)
    energy = audio.pow(2)

    # find indices above threshold
    indices = (energy > threshold).nonzero(as_tuple=True)[0]

    # if completely silent, return original
    if indices.numel() == 0:
        return audio
    
    # convert ms to samples
    pad_samples = int((pad_ms / 1000) * sample_rate)
    min_samples = int((min_dur_ms / 1000) * sample_rate)

    start_idx = max(0, indices[0].item() - pad_samples)
    end_idx = min(audio.shape[0], indices[-1].item() + 1 + pad_samples)

    trimmed_audio = audio[start_idx : end_idx]

    if trimmed_audio.shape[0] < min_samples:
        return 
    
    return trimmed_audio

def get_mel_from_wav(audio, _stft):
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).clip(-1, 1)
    audio = audio.requires_grad_(False)
    mel_spec, energy = _stft.mel_spectrogram(audio)
    mel_spec = mel_spec.squeeze(0).cpu().numpy().astype(np.float32)
    energy = energy.squeeze(0).cpu().numpy().astype(np.float32)

    return mel_spec, energy
