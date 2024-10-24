from models.model import SegmentShufflenetV2
from data.data_loader import data_loader
from utils.args import get_args
import torchmetrics
from tqdm import tqdm
from utils.matrix import binary_evaluate_metrics, multiclass_evaluate_metrics
from utils.utils import save_ckpt, save_result, create_ckpt

import torch.quantization as quant
import torch

def train():
    torch.cuda.empty_cache()

    args = get_args()
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    ckpt_dir = './ckpt'
    
    train_loader = data_loader(image_dir, mask_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    model = args.models(args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    for param in model.parameters():
        param.requires_grad = True

    if args.weight:
        model.load_state_dict(torch.load(args.weight, map_location=device, weights_only=True))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        running_loss = 0.
        metrics = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0, 'Dice': 0, 'IoU': 0}
        metric_count = 0

        if epoch == 1:
            print('start training!')
            print(f'cuda: {torch.cuda.is_available()}')

        if epoch == args.epochs:
            output_dir = create_ckpt('./output')


        for idx, (image, label) in tqdm(enumerate(train_loader)):
            image, label = image.to(device).to(torch.float32), label.to(device).to(torch.long)

            output = model(image)

            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            output = torch.argmax(output, dim = 1)

            batch_metrics = multiclass_evaluate_metrics(output, label, device, acc = True, pre = True, re = True, io = True)
       
            for metric_name, value in batch_metrics.items():
                metrics[metric_name] += value
            metric_count += 1

            if epoch == args.epochs:
                save_result(output, file_dir = output_dir)
        
        scheduler.step()

        print(f'epoch: {epoch} / {args.epochs}, loss: {running_loss / len(train_loader)}')
        for metric_name, value in metrics.items():
            metrics[metric_name] /= metric_count


        for metric, value in metrics.items():
            print(f'{metric}: {value}')

        if epoch % 5 == 0:
            if epoch == 5:
                ckpt_dir = create_ckpt()
            save_ckpt(model, epoch, ckpt_dir)

            
if __name__ == '__main__':
    train()