from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
from tqdm import tqdm as progress_bar
from torch import nn
from dataloader import get_dataloader
from torch.cuda.amp import GradScaler, autocast

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


def baseline_train(args, vae, clip_tokenizer, unet_model, datasets, device):
    criterion = nn.MSELoss()
    train_dataloader = get_dataloader(datasets['train'], args["batch_size"])
    
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=args["learning_rate"], eps=args["adam_epsilon"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["num_epochs"])
    noise_scheduler = DDPMScheduler(num_train_timesteps=args["num_timesteps"])

    scaler = GradScaler()
    for epoch_count in range(args["num_epochs"]):
        losses = 0
        unet_model.train()
        #with memory_summary(device=device, abbreviated=True):
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            
            #optimizer.zero_grad()
            # Need to get images of size (batch_size x n_channels x height x width)
            # Need to get texts of size (batch_size x n_sequence)

            #images, texts = prepare_inputs(batch)
            texts, images = batch
            #print(images.size())  # or images.shape
            images = images.to(device)

            batch_size = images.shape[0]
            # Gradients do not flow into the autoencoder or the transformer
            # Option 1 (below) / Option 2: Make requires grad of all the parameters of VAE and tokenizer as False (not implemented) 
            with torch.no_grad(): 

                # Convert texts to embeddings (batch_size x n_sequence x n_embed)
                text_embeddings = clip_tokenizer(texts).to(device)
                # Convert image into latent images
                clean_img_latents = vae.encode(images).latent_dist.sample()
            
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device,
                dtype=torch.int64
            )

            noise = torch.randn(clean_img_latents.shape, device=device)

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_img_latents = noise_scheduler.add_noise(clean_img_latents, noise, timesteps).to(device=device)
            with autocast():
                noise_pred = unet_model(noisy_img_latents, text_embeddings, timesteps)
                loss = criterion(noise_pred.sample, noise)
            scaler.scale(loss).backward()
            if (step + 1) % args["grad_acc_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                print("Performing gradient accumulation")
            
            losses += loss.item()
        lr_scheduler.step()

        # Commenting out running of evaluation of the 
        #run_eval(args, model, datasets, tokenizer, 'validation')
        print('epoch', epoch_count, '| losses:', losses)

    return unet_model