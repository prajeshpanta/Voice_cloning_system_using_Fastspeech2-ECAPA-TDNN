import torch

"""this code is used to check the properties of the embedding tensor, such as its shape, min, max, mean, std, and norm."""
device="cpu"
tensor = torch.load('embeddings/LibriTTS/5.pt', map_location=device)

print("Shape:", tensor.shape)
print("Min:", torch.min(tensor).item())
print("Max:", torch.max(tensor).item())
print("Mean:", torch.mean(tensor).item())
print("Std:", torch.std(tensor).item())
print(tensor.norm())