import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import  AutoencoderKL
from model import Diffusion,CLIPTextEmbedder

if(torch.backends.mps.is_available()):
    device = torch.device("mps")
elif(torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = "cpu"

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to(device)
tokenizer = CLIPTextEmbedder()


