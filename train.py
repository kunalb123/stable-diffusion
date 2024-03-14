from diffusers import UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from tqdm import tqdm as progress_bar
from torch import nn


def add_gaussian_noise(images, mean=0.0, std=0.1):
    """Adds Gaussian noise to a tensor of images."""
    noise = torch.randn_like(images) * std + mean
    return images + noise

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
    criterion = nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(args, datasets['train'])
    
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    noise_scheduler = # add in the noise schehduler


    for epoch_count in range(args.max_epochs):
        losses = 0
        unet_model.train()
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):

            # Need to get images of size (batch_size x n_channels x height x width)
            # Need to get texts of size (batch_size x n_sequence)

            inputs, texts = prepare_inputs(batch) 
            
            # Convert texts to embeddings (batch_size x n_sequence x n_embed)
            

            # Convert image into latent images

            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()
            model.scheduler.step()
            model.zero_grad()
            losses += loss.item()
    
        run_eval(args, model, datasets, tokenizer, 'validation')
        print('epoch', epoch_count, '| losses:', losses)

if __name__ == '__main__':
    baseline_train(None, None, None, None)
