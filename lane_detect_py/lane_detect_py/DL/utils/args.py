import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    # model.py
    parser.add_argument('--models', type = str, default = 'Unet', help = 'decide model name')
    parser.add_argument('--num_classes', type=int, default = 5, help='Number of classes')

    # data_loader.py
    parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size for train')
    parser.add_argument('--num_workers', type = int, default = 2, help = 'Num_workers')
    parser.add_argument('--image_dir', type=str, default='./data/img/', help = 'train_image_directory')
    parser.add_argument('--mask_dir', type=str, default='./data/mask/', help = 'train_mask_directory')
    
    # train.py
    parser.add_argument('--epochs', type=int, default = 100, help = 'num_epochs')
    parser.add_argument('--lr', type=float, default = 0.001, help = 'set learning rate')

    # inference
    parser.add_argument('--threshold', type=float, default = 0.5, help = 'evaluate threshold')
    parser.add_argument('--weight', type=str, help = 'load pretrained weight')
    
    return parser.parse_args()