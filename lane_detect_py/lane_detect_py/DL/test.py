import cv2
import numpy as np
import random
import os

def label_modify(img_dir, label_dir, save_dir):
    img_dir = img_dir
    label_dir = label_dir
    
    img_list = os.listdir(img_dir)
    img_list = sorted(img_list)

    label_list = os.listdir(label_dir)
    label_list = sorted(label_list)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for label in label_list:
        mask = cv2.imread(label_dir + label, cv2.IMREAD_GRAYSCALE)

        mask = np.where(mask == 2, 0, 1).astype(np.uint8)
        save = os.path.join(save_dir, label)
        cv2.imwrite(save, mask)

def shape_check(img_dir, label_dir):
    img_dir = img_dir
    label_dir = label_dir
    
    img_list = os.listdir(img_dir)
    img_list = sorted(img_list)

    label_list = os.listdir(label_dir)
    label_list = sorted(label_list)

    for image in img_list:
        image = cv2.imread(img_dir + image)
        print(image.shape)


def dummy_delete(image_dir, label_dir):    
    img_list = os.listdir(image_dir)
    img_list = sorted(img_list)

    label_list = os.listdir(label_dir)
    label_list = sorted(label_list)

    for img in img_list:
        if 'Zone.' in img:
            img_path = os.path.join(image_dir, img)
            os.remove(img_path)

    for label in label_list:
        if 'Zone.' in label:
            label_path = os.path.join(label_dir, label)
            os.remove(label_path)

def show(image_dir = './test_data/image/', label_dir = './test_data/new_mask/'):
    img_list = os.listdir(image_dir)
    img_list = sorted(img_list)

    label_list = os.listdir(label_dir)
    label_list = sorted(label_list)

    while True:
        rand = random.randint(0, len(img_list) - 1)
        img = os.path.join(image_dir, img_list[rand])
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = os.path.join(label_dir, label_list[rand])
        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        
        label = np.where(label == 1, 255, 0)
        label = cv2.convertScaleAbs(label)
        cv2.imshow('img', img)
        cv2.imshow('label', label)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        break
        

if __name__ == '__main__':
    # dummy_delete('./test_data/image/', './test_data/mask/')
    # label_modify('./test_data/image/', './test_data/mask/', './test_data/new_mask/')
    # shape_check('./test_data/image/', './test_data/mask/')
    show()