from lane_detect_py.DL.models.model import Unet
from lane_detect_py.DL.utils.args import get_args
import torch
from lane_detect_py.DL.data.data_loader import transform
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image as PILImage

def inference(frame = 'data/img/image199.jpg'):
    new_frame = PILImage.fromarray(frame)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Unet(5).to(device)
    model.load_state_dict(torch.load('/home/gyesu96/lane_detect/src/self_driving/lane_detect_py/lane_detect_py/DL/ckpt/model_epoch_100', map_location=device, weights_only=True))

    model.eval()

    with torch.no_grad():
        to_tensor = transforms.Compose([
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize([112.014, 111.990, 108.771], [72.391, 72.396, 70.402]),
            ])
        
        image = to_tensor(new_frame).unsqueeze(0)
        image = image.to(device).float()
        output = model(image)
        output = torch.argmax(output, dim = 1)
        output = output.squeeze(0).squeeze(0).detach().cpu().numpy()
        # output = np.where(output > 0.5, 255, 0).astype('uint8')
        output = visualize_mask(output)

    return output, frame

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

if __name__ == '__main__':
    output, frame = inference()
    frame = np.array(frame)
    print(np.unique(output))
    cv2.imshow('origin', frame[:,:,0])

    cv2.imshow('inference', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()