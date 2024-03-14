from diffusers import UNet2DConditionModel
import torch
from tqdm import tqdm as progress_bar
from torch import nn

NUM_EPOCHS = 10

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
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'])
    
    # task2: setup model's optimizer_scheduler if you have
    # model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    model.optimizer = custom_optimizer(model, base_lr=args.learning_rate, decay_rate=0.95)
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, 10)
    # model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=1, gamma=0.1)
    
    # task3: write a training loop
    for epoch_count in range(NUM_EPOCHS):
        losses = 0
        model.train()
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()
    
        run_eval(args, model, datasets, tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)
  

if __name__ == '__main__':
    baseline_train(None, None, None, None)