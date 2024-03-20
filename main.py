import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import  AutoencoderKL
import yaml
from datasets import load_dataset
from dataloader import get_dataloader
from train import baseline_train, inference
from model import Diffusion, CLIPTextEmbedder

if(torch.backends.mps.is_available()):
    device = torch.device("mps")
elif(torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = "cpu"

def print_memory_usage(description):
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    print(f"{description}: Allocated: {allocated / 1e9:.2f} GB, Cached: {cached / 1e9:.2f} GB")

print_memory_usage('Begin')
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float).to(device)
tokenizer = CLIPTextEmbedder(device=device)
print_memory_usage('Loaded VAE, CLIP')


def main():
    with(open("config.yaml", "r")) as file:
        config = yaml.safe_load(file)
    
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_timesteps = config["num_timesteps"]

    print("Using parameters:\nnum_epochs:", num_epochs, "\nbatch_size:", batch_size, "\nlearning_rate:", learning_rate, "\nnum_timesteps:", num_timesteps)

    print("Loading the dataset...")
    # Load the dataset with the `large_random_1k` subset
    dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k')
    print("Dataset loaded")

    print("Training the model...")
    unet_model = Diffusion().to(device)
    print_memory_usage('Loaded UNET')    
    unet_model = baseline_train(config, vae, tokenizer, unet_model, dataset, device)

    # Save the model
    torch.save(unet_model.state_dict(), "model.pth")
    print("Model trained")

    # Inference
    print("Inference...")
    prompts = ["a small liquid sculpture, corvette, viscous, reflective, digital art", "human sculpture of lanky tall alien on a romantic date at italian restaurant with smiling woman, nice restaurant, photography, bokeh"]
    inference(config, vae, tokenizer, unet_model, prompts, dataset, device)



if __name__ == "__main__":
    main()