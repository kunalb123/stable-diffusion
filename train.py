from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
from tqdm import tqdm as progress_bar
from torch import nn
from dataloader import get_dataloader

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)
        
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
  
    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))


def baseline_train(args, vae, clip_tokenizer, unet_model, datasets):
    criterion = nn.MSELoss()
    train_dataloader = get_dataloader(args, datasets['train'])
    
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps)


    for epoch_count in range(args.max_epochs):
        losses = 0
        unet_model.train()
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            
            optimizer.zero_grad()
            # Need to get images of size (batch_size x n_channels x height x width)
            # Need to get texts of size (batch_size x n_sequence)

            images, texts = prepare_inputs(batch)

            batch_size = images.shape[0]
            # Gradients do not flow into the autoencoder or the transformer
            # Option 1 (below) / Option 2: Make requires grad of all the parameters of VAE and tokenizer as False (not implemented) 
            with torch.no_grad(): 

                # Convert texts to embeddings (batch_size x n_sequence x n_embed)
                text_embeddings = clip_tokenizer(texts)
                # Convert image into latent images
                clean_img_latents = vae.encode(images)
            
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=clean_img_latents.device,
                dtype=torch.int64
            )

            noise = torch.randn(clean_img_latents.shape, device=clean_img_latents.device)

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_img_latents = noise_scheduler.add_noise(clean_img_latents, noise, timesteps)

            noise_pred = unet_model(noisy_img_latents,text_embeddings,timesteps)
            loss = criterion(noise_pred,noise)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            
            losses += loss.item()

        # Commenting out running of evaluation of the 
        #run_eval(args, model, datasets, tokenizer, 'validation')
        print('epoch', epoch_count, '| losses:', losses)

