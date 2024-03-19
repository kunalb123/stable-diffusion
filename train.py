from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
from tqdm import tqdm as progress_bar
from torch import nn
from dataloader import get_dataloader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from PIL import Image

def print_memory_usage(description):
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    print(f"{description}: Allocated: {allocated / 1e9:.2f} GB, Cached: {cached / 1e9:.2f} GB")

def add_gaussian_noise(images, mean=0.0, std=0.1):
    """Adds Gaussian noise to a tensor of images."""
    noise = torch.randn_like(images) * std + mean
    return images + noise

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

#     acc = 0
#     for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
#         inputs, labels = prepare_inputs(batch, model)
#         logits = model(inputs, labels)
        
#         tem = (logits.argmax(1) == labels).float().sum()
#         acc += tem.item()
  
#     print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))


def baseline_train(args, vae, clip_tokenizer, unet_model, train_dataloader, device):
    criterion = nn.MSELoss()
    train_dataloader = get_dataloader(train_dataloader['train'], args["batch_size"])
    
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=args["learning_rate"], eps=args["adam_epsilon"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["num_epochs"])
    noise_scheduler = DDPMScheduler(num_train_timesteps=args["num_timesteps"])

    scaler = GradScaler()
    for epoch_count in range(args["num_epochs"]):
        # print_memory_usage("Start of Epoch")
        losses = 0
        unet_model.train()
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            # print_memory_usage(f"Before Step {step}")
            optimizer.zero_grad()
            # Need to get images of size (batch_size x n_channels x height x width)
            # Need to get texts of size (batch_size x n_sequence)

            #images, texts = prepare_inputs(batch)
            texts, images = batch
            images = images.to(device)

            batch_size = images.shape[0]
            # Gradients do not flow into the autoencoder or the transformer
            # Option 1 (below) / Option 2: Make requires grad of all the parameters of VAE and tokenizer as False (not implemented) 
            with torch.no_grad(): 

                # Convert texts to embeddings (batch_size x n_sequence x n_embed)
                texts = clip_tokenizer(texts).to(device)
                # Convert image into latent images
                images = vae.encode(images).latent_dist.sample()
            
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device,
                dtype=torch.int64
            )

            noise = torch.randn(images.shape, device=device)

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            images = noise_scheduler.add_noise(images, noise, timesteps).to(device=device)
            with autocast():
                noise_pred = unet_model(images, texts, timesteps)
                loss = criterion(noise_pred.sample, noise)
            scaler.scale(loss).backward()
            # print(f'loss = {loss}')
            # print_memory_usage(f"After Backward {step}")
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            # print_memory_usage(f"After Step {step}")
            losses += loss.item()


        # Commenting out running of evaluation of the 
        #run_eval(args, model, datasets, tokenizer, 'validation')
        print('epoch', epoch_count, '| losses:', losses)

    return unet_model

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
                latent_image = unet_model(latent_image, text_embeddings, timestep_tensor)
        
        # Decode the generated latent image back to image space
        with torch.no_grad():
            generated_image = vae.decode(latent_image).sample()
        
        # Postprocess and return the generated image
        generated_image = ((generated_image.squeeze(0).add(1).div(2).clamp(0, 1)) * 255).cpu().numpy().astype(np.uint8)
        generated_image = np.transpose(generated_image, (1, 2, 0))  # Convert from CHW to HWC format for PIL
        pil_image = Image.fromarray(generated_image)
        pil_image.save(save_path)
        print(f"Image saved to {save_path}")
    
    return