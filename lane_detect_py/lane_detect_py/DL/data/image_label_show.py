import cv2
import os
import numpy as np
import random


def show(image_dir = '../test_data/image/', label_dir = '../test_data/new_mask/'):
    img_list = os.listdir(image_dir)
    img_list = sorted(img_list)

    label_list = os.listdir(label_dir)
    label_list = sorted(label_list)

    while True:
        rand = random.randint(0, len(img_list) - 1)
        if 'aug' in img_list[rand]:
            img = os.path.join(image_dir, img_list[rand])
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            label = os.path.join(label_dir, label_list[rand])
            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            print(np.unique(label))
            color_label = cv2.applyColorMap(label, cv2.COLORMAP_JET)
            cv2.imshow('img', img)
            cv2.imshow('label', color_label)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            break

if __name__ == '__main__':
    show()