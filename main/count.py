from config import cfg
from nets.network import get_model


model = get_model("train")
print(f'[INFO] Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
