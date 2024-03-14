from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
import torch
import torch.nn as nn

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

    def __init__(self, config):
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16)
        self.unet = UNet2DConditionModel()
        self.tokenizer = CLIPTextEmbedder()

    def forward(self, image, text, timestep):
        # add gaussian noise to image every forward pass
        gaussian_noise = torch.randn_like(image)
        noisy_image = image + gaussian_noise

        latent_image = self.vae.encode(noisy_image)
        latent_text = self.tokenizer(text)
        return self.unet(latent_image, latent_text, timestep)
    

clip = CLIPTextEmbedder()
prompt = ["a photo of a cat", "a photo of a dog"]
print(clip(prompt).shape)