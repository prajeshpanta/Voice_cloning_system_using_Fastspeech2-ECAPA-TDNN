import yaml
import librosa
import numpy as np
from embeddinggenerator import generate_embedding
import torch
import pyworld as pw
from scipy.interpolate import interp1d
from audio.stft import TacotronSTFT
from audio.tools import get_mel_from_wav

def compute_pitch(wav, sampling_rate=22050, hop_length=256):
    # compute fundamental frequency
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sampling_rate,
        frame_period=hop_length / sampling_rate * 1000
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)

    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    pitch = interp_fn(np.arange(0, len(pitch)))

    return pitch

def cosine_similarity_pct(e1: torch.Tensor, e2: torch.Tensor) -> float:
    e1 = torch.nn.functional.normalize(e1.view(1, -1).float(), p=2, dim=1)
    e2 = torch.nn.functional.normalize(e2.view(1, -1).float(), p=2, dim=1)
    cos = float((e1 * e2).sum())
    return round((cos + 1) / 2 * 100, 2)

def compute_overall_similarity(mel1, mel2, energy1, energy2, pitch1, pitch2):
    # Compute cosine similarity for mel spectrograms
    mel_similarity = np.dot(mel1.flatten(), mel2.flatten()) / (
        np.linalg.norm(mel1.flatten()) * np.linalg.norm(mel2.flatten())
    )
    
    # Compute mean absolute error for energy
    energy_similarity = 1 - np.mean(np.abs(energy1 - energy2) / (np.max(energy1) + 1e-8))
    
    # Compute mean absolute error for pitch
    pitch_similarity = 1 - np.mean(np.abs(pitch1 - pitch2) / (np.max(pitch1) + 1e-8))
    
    # # Combine similarities (you can adjust weights as needed)
    # overall_similarity = (mel_similarity + energy_similarity + pitch_similarity) / 3
    
    return  mel_similarity, energy_similarity, pitch_similarity



if __name__=="__main__":
    preprocess_config_path = "config/preprocess.yaml"
    with open(preprocess_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    # path
    wav=librosa.load("similarity/original/8887_281471_000000_000000.wav", sr=22050, mono=True)
    original_embedding=torch.load("similarity/original/8887.pt")
    synth=librosa.load("similarity/synthesized/8887_281471_000000_000000.wav", sr=22050, mono=True)
    synth_embedding=generate_embedding("similarity/synthesized/8887_281471_000000_000000.wav","similarity/synthesized/8887_synth.pt")
    synth_embedding=torch.load("similarity/synthesized/8887_synth.pt")



    mel1, energy1 = get_mel_from_wav(wav, STFT) 
    pitch1=compute_pitch(wav) 

    mel2, energy2 = get_mel_from_wav(synth, STFT)   
    pitch2=compute_pitch(synth)

    a,b,c=compute_overall_similarity(mel1, mel2, energy1, energy2, pitch1, pitch2)
    embedding_similarity=cosine_similarity_pct(original_embedding, synth_embedding)


    print("Mel Similarity:", a)
    print("Energy Similarity:", b)
    print("Pitch Similarity:", c)
    print("Embedding Similarity:", embedding_similarity)
    