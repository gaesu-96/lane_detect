import cv2
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def create_ckpt(ckpt_dir = './ckpt'):
    i = 1
    while True:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            return ckpt_dir
        else:
            if i == 1:
                temp = ckpt_dir
            idx = str(i)
            ckpt_dir = temp + idx
            i += 1

def save_ckpt(model, epoch, ckpt_dir = './ckpt'):
    torch.save(model.state_dict(), f'{ckpt_dir}/model_epoch_{epoch}')

def save_result(tensor, file_dir = './output'):
    batch_size, height, width = tensor.size()

    # if channels == 1:
    #     images = tensor.squeeze(1)
    # else:
    #     images = torch.argmax(tensor, dim = 1)
    images = tensor.detach().cpu().numpy()

    if len(images.shape) == 3:
        for idx, batch in enumerate(images):
            mask = visualize_mask(batch)
            cv2.imwrite(file_dir + '/' + str(idx) + '.jpg', mask)



def visualize_mask(mask):
    color_map = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0)
    }

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for label, color in color_map.items():
        color_mask[mask == label] = color

    return color_mask
        
    