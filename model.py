from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as tfms  

if(torch.backends.mps.is_available()):
    device = torch.device("mps")
elif(torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = "cpu"
class CLIPTextEmbedder(nn.Module):

    def __init__(self, version: str = "openai/clip-vit-large-patch14", device="cuda:0", max_length: int = 77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version, torch_dtype=torch.float16)
        self.transformer = CLIPTextModel.from_pretrained(version, torch_dtype=torch.float16).to(device)
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
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to(device)
        self.unet = UNet2DConditionModel().to(device)
        self.tokenizer = CLIPTextEmbedder()

    def forward(self, image, text, timestep):
        # add gaussian noise to image every forward pass
        gaussian_noise = torch.randn_like(image)
        noisy_image = image + gaussian_noise

        latent_image = self.vae.encode(noisy_image)
        latent_text = self.tokenizer(text)
        return self.unet(latent_image, latent_text, timestep)
    

image = Image.open("image.png").convert("RGB").resize((256, 256))
image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
image = image.to(device, dtype=torch.float16)
diffuser = Diffusion()
result = diffuser.forward(image, torch.tensor(40, dtype=torch.float16, device=device), ["sattelite image of a city"])
print(result)