from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import os
import torch
import numpy as np

class DrivingDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform = None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_list = os.listdir(self.img_dir)
        self.image_list = sorted(self.image_list)
        self.label_list = os.listdir(self.label_dir)
        self.label_list = sorted(self.label_list)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        try:
            if self.image_list[index] is None or self.label_list is None:
                print(f'image/label is None! : {index}')
                pass
            image = os.path.join(self.img_dir, self.image_list[index])
            image = Image.open(image)

            if self.transform:
                image = self.transform(image)
            else:
                image = np.array(image)
                image = torch.tensor(image)
                image = torch.permute(image, (2, 0, 1))

            label = os.path.join(self.label_dir, self.label_list[index])
            label = Image.open(label)
            label_transform = transforms.Resize((720, 1280))
            label = label_transform(label)
            image = label_transform(image)

            label = np.array(label)
            label = torch.tensor(label)

            return image, label

        except Exception as e:
            return self.__getitem__((index + 1) % len(self))

def normalization(dataset, origin = True):
    if origin:
        image = np.array([img.numpy() for img, _ in dataset if img.shape == (3, 720, 1280)])
    else:
        image = np.array([img.numpy() for img, _ in dataset])

    mean_r = np.mean(image, axis = (2, 3))[:, 0].mean()
    mean_g = np.mean(image, axis = (2, 3))[:, 1].mean()
    mean_b = np.mean(image, axis = (2, 3))[:, 2].mean()

    std_r = np.std(image, axis = (2, 3))[:, 0].mean()
    std_g = np.std(image, axis = (2, 3))[:, 1].mean()
    std_b = np.std(image, axis = (2, 3))[:, 2].mean()

    return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]

def batch_normalization(data_loader):
    # Initialize sums and squares for each channel
    mean_r, mean_g, mean_b = 0, 0, 0
    std_r, std_g, std_b = 0, 0, 0

    for images, _ in data_loader:
        # Compute sum for each batch
        images = images.numpy()
        images = images.reshape(images.shape[0], images.shape[1], -1)  # Flatten the HxW dimension
        mean_r += np.mean(images, axis = 2)[:, 0].mean()
        mean_g += np.mean(images, axis = 2)[:, 1].mean()
        mean_b += np.mean(images, axis = 2)[:, 2].mean()

        std_r += np.std(images, axis = 2)[:, 0].mean()
        std_g += np.std(images, axis = 2)[:, 1].mean()
        std_b += np.std(images, axis = 2)[:, 2].mean()

    # Calculate final mean and standard deviation
    mean_r /= len(data_loader)
    mean_g /= len(data_loader)
    mean_b /= len(data_loader)

    std_r /= len(data_loader)
    std_g /= len(data_loader)
    std_b /= len(data_loader)

    return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]


def transform(mean, std):
    return transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])

def data_loader(image_dir, label_dir, batch_size, num_workers):
    print('dataset loading start!')
    dataset = DrivingDataset(image_dir, label_dir)
    data_loader = DataLoader(dataset, batch_size = 16, shuffle=False, num_workers=num_workers)
    print('start normalization!')
    mean, std = batch_normalization(data_loader)
    print(f'data normalization finished! \nmean:{mean}, \nstd: {std}')
    transformer = transform(mean, std)
    dataset = DrivingDataset(image_dir, label_dir, transform=transformer)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)
    print('data loading finish!')
    return data_loader