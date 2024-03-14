from diffusers import UNet2DConditionModel
import torch
from tqdm import tqdm as progress_bar
from torch import nn

NUM_EPOCHS = 10

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


def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(args, datasets['train'])
    
    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=10)

    initial_std = 0.1
    std_increase = 0.001

    for epoch_count in range(NUM_EPOCHS):
        losses = 0
        model.train()
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            
            # Adding noise to inputs: assuming theese are images
            inputs = add_gaussian_noise(inputs, std=initial_std + step*std_increase)

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
