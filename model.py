from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as tfms  

if(torch.backends.mps.is_available()):
    device = torch.device("mps")
elif(torch.cuda.is_available()):
    num_devices = torch.cuda.device_count()
    if num_devices >= 8:
        device = torch.device("cuda:6")  # Choose device index 6 (zero-based indexing)
    else:
        device = torch.device("cuda:0")  # Default to device index 0 if fewer than 8 devices are available
else:
    device = "cpu"

    
class CLIPTextEmbedder(nn.Module):

    def __init__(self, version: str = "openai/clip-vit-base-patch32", device=device, max_length: int = 77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version, torch_dtype=torch.float)
        self.transformer = CLIPTextModel.from_pretrained(version, torch_dtype=torch.float).to(device)
        self.device = device
        self.max_length = max_length

    def forward(self, prompts: list[str]):
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return self.transformer(input_ids=tokens).last_hidden_state

class Diffusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel(cross_attention_dim=512).to(device)

    def forward(self, latent_image, latent_text, timestep):
        # add gaussian noise to image every forward pass
        return self.unet(sample=latent_image, encoder_hidden_states=latent_text, timestep=timestep)
    

# Example usage

# image = Image.open("image.png").convert("RGB").resize((256, 256))
# image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
# image = image.to(device, dtype=torch.float)
# vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float).to(device)
# tokenizer = CLIPTextEmbedder()
# text = ["sattelite image of a city"]
# latent_image = vae.encode(image).latent_dist.sample()
# latent_text = tokenizer(text).to(device)
# timestep = torch.tensor(40, dtype=torch.int, device=device)
# diffuser = Diffusion()
# result = diffuser.forward(latent_image, latent_text, timestep)
# print(result)