import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from PIL import Image
import numpy as np
from tqdm import tqdm as progress_bar
from diffusers import  AutoencoderKL
from model import Diffusion, CLIPTextEmbedder
from transformers import CLIPTextModel, CLIPTokenizer, logging
import yaml



def inference(args, vae, clip_encoder, unet_model, prompts, save_path, device):
    noise_scheduler = DDPMScheduler(num_train_timesteps=args["num_timesteps"], beta_schedule='squaredcos_cap_v2')
    guidance_scale = 7.5 

    #unet_model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)

    for step, prompt in progress_bar(enumerate(prompts), total=len(prompts)):

        latent_image = torch.randn((1, 4, 64, 64), device=device)
        latent_image = latent_image * noise_scheduler.init_noise_sigma
        
        # Run the diffusion process
        for i, t in progress_bar(enumerate(noise_scheduler.timesteps), total=len(noise_scheduler.timesteps)):

            latent_model_input = torch.cat([latent_image] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            # Generate image from text embedding and latent image
            with torch.no_grad():
                text_embeddings = clip_encoder(prompt).to(device)
                noise_pred = unet_model(sample=latent_model_input, encoder_hidden_states=text_embeddings, timestep=t).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latent_image = noise_scheduler.step(noise_pred, t, latent_image).prev_sample
        # Decode the generated latent image back to image space
        latent_image = 1 / 0.18215 * latent_image

        with torch.no_grad():
            generated_image = vae.decode(latent_image).sample
        
        # Postprocess and return the generated image
        generated_image = ((generated_image.squeeze(0).add(1).div(2).clamp(0, 1)) * 255).cpu().numpy().astype(np.uint8)
        generated_image = np.transpose(generated_image, (1, 2, 0))  # Convert from CHW to HWC format for PIL
        pil_image = Image.fromarray(generated_image)
        pil_image.save(save_path+str(step)+".png")
        print(f"Image saved to {save_path+str(step)}"+".png")
    
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

    #unet_model = UNet2DConditionModel.from_pretrained(
    #"CompVis/stable-diffusion-v1-4", subfolder="unet",use_safetensors=True).to(device)
    #unet_model.eval()

    prompts = ["A watercolor painting of an otter"]
    inference(config, vae, tokenizer, unet_model, prompts, '200epochmodel', device)

if __name__ == "__main__":
    main()