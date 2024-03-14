import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import  AutoencoderKL
import yaml
from datasets import load_dataset
from dataloader import get_dataloader
from train import baseline_train
from model import Diffusion, CLIPTextEmbedder

if(torch.backends.mps.is_available()):
    device = torch.device("mps")
elif(torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = "cpu"

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to(device)
tokenizer = CLIPTextEmbedder()

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
    unet_model = baseline_train(config, vae, tokenizer, unet_model, dataset)
    
    # Save the model
    torch.save(unet_model.state_dict(), "model.pth")
    print("Model trained")

    # Inference
    print("Inference...")
    # dataloader = get_dataloader(dataset, batch_size, "validation")


if __name__ == "__main__":
    main()