import cv2
import numpy as np
import random
import os

class Augmentation():
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        
        self.img_list = os.listdir(self.img_dir)
        self.img_list = sorted(self.img_list)

        self.label_list = os.listdir(self.label_dir)
        self.label_list = sorted(self.label_list)

        self.aug_list = [
            self.brightness,
            self.rotation,
            self.flip,
            self.blur,
            self.crop
        ]

    def augmentation(self):
        for i in range(len(self.img_list)):
            image = os.path.join(self.img_dir, self.img_list[i])
            label = os.path.join(self.label_dir, self.label_list[i])

            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)

            augs = random.sample(self.aug_list, 2)

            for aug in augs:
                image, label = aug(image, label)

            aug_image_name = os.path.join(self.img_dir, f'aug_{self.img_list[i]}')
            aug_label_name = os.path.join(self.label_dir, f'aug_{self.label_list[i]}')

            cv2.imwrite(aug_image_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(aug_label_name, label)

    def brightness(self, image, label, min = 30, max = 100):
        brightness = random.randint(min, max)

        modify_img = np.clip(image + brightness, 0, 255)

        return modify_img, label
    
    def rotation(self, image, label, ang = 180):
        angle = random.randint(-ang, ang)
        h, w = image.shape[:2]
        modify = cv2.getRotationMatrix2D((h//2, w//2), angle, 1)
        rotation_img = cv2.warpAffine(image, modify, (w, h))
        rotation_label = cv2.warpAffine(label, modify, (w, h))

        return rotation_img, rotation_label

    def vertical_flip(self, image, label):
        flipped_image = cv2.flip(image, 1)
        flipped_label = cv2.flip(label, 1)
        
        return flipped_image, flipped_label
    
    def horizon_flip(self, image, label):
        flipped_image = cv2.flip(image, 0)
        flipped_label = cv2.flip(label, 0)

        return flipped_image, flipped_label
    
    def vert_horizon_flip(self, image, label):
        flipped_image = cv2.flip(image, -1)
        flipped_label = cv2.flip(label, -1)

        return flipped_image, flipped_label
    
    def flip(self, image, label):
        flip_func = [
            self.vertical_flip,
            self.horizon_flip,
            self.vert_horizon_flip
            ]
        rand = random.choice(flip_func)

        flipped_image, flipped_label = rand(image, label)

        return flipped_image, flipped_label
    
    def blur(self, image, label):
        sigma = random.randint(1, 10)
        blurred_image = cv2.GaussianBlur(image, (5,5), sigma)

        return blurred_image, label
    
    def crop(self, image, label):
        y, x = image.shape[:2]
        point_x = random.randint(0, x - 301)
        point_y = random.randint(0, y - 201)

        cropped_image = image[point_y:point_y + 200, point_x:point_x + 300, :]
        cropped_label = label[point_y:point_y + 200, point_x:point_x + 300]

        return cropped_image, cropped_label
    
if __name__ == '__main__':
    print('aug started!')

    image_dir = './img/'
    label_dir = './mask/'

    aug = Augmentation(image_dir, label_dir)
    aug.augmentation()
    new_image = os.listdir(image_dir)
    new_label = os.listdir(label_dir)


    print(f'aug_finish! \nimage_count: {len(new_image)} \nlabel_count: {len(new_label)}')