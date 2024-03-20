import torch
from PIL import Image
import numpy as np
from tqdm import tqdm as progress_bar
from diffusers import  AutoencoderKL
from model import Diffusion, CLIPTextEmbedder
import yaml


def inference(args, vae, clip_encoder, unet_model, prompts, save_path, device):

    for step, prompt in progress_bar(enumerate(prompts), total=len(prompts)):
        latent_image = torch.randn((1, 4, 64, 64), device=device)
        
        # Run the diffusion process
        for timestep in reversed(range(args["num_timesteps"])):
            # Convert timestep to tensor
            timestep_tensor = torch.tensor([timestep] * latent_image.shape[0], dtype=torch.long, device=device)
            
            # Generate image from text embedding and latent image
            with torch.no_grad():
                text_embeddings = clip_encoder(prompt).to(device)
                latent_image = unet_model(latent_image, text_embeddings, timestep_tensor).sample
        
        # Decode the generated latent image back to image space
        with torch.no_grad():
            generated_image = vae.decode(latent_image).sample
        
        # Postprocess and return the generated image
        generated_image = ((generated_image.squeeze(0).add(1).div(2).clamp(0, 1)) * 255).cpu().numpy().astype(np.uint8)
        generated_image = np.transpose(generated_image, (1, 2, 0))  # Convert from CHW to HWC format for PIL
        pil_image = Image.fromarray(generated_image)
        pil_image.save(save_path+str(step)+".png")
        print(f"Image saved to {save_path+str(step)+".png"}")
    
    return

def main():
    with(open("config.yaml", "r")) as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float).to(device)
    tokenizer = CLIPTextEmbedder(device=device)
    unet_model = Diffusion().to(device)
    unet_model.load_state_dict(torch.load("model.pth"))
    unet_model.eval()
    prompts = ["a small liquid sculpture, corvette, viscous, reflective, digital art", "human sculpture of lanky tall alien on a romantic date at italian restaurant with smiling woman, nice restaurant, photography, bokeh"]
    inference(config, vae, tokenizer, unet_model, prompts, '200epochmodel', device)

if __name__ == "__main__":
    main()