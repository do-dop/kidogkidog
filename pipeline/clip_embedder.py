import clip
import torch
import torch.nn.functional as F
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


def embed_image(image_path):
    """이미지 → CLIP 벡터"""
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(img)
        embedding = F.normalize(embedding, dim=-1)
    return embedding.cpu().numpy().tolist()[0]  # list로 반환


def embed_text(text):
    """텍스트 → CLIP 벡터"""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = F.normalize(embedding, dim=-1)
    return embedding.cpu().numpy().tolist()[0]  # list로 반환
