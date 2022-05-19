from http.client import IM_USED
import os
import cv2
import torch
from PIL import Image


def load_dataset(path, transforms):
    dataset = []
    for file in os.listdir(path):
        # Load images and their class into features and targets
        image = Image.open(path + file)
        image = image.convert('RGB')
        if image is not None:
            image = transforms(image)
            print('Loading:', file, '|', 'Shape:', image.shape)
            dataset.append((image, 0))
    return dataset


def save_dataset(dataset, path):
    size = dataset.shape[0]
    for i in range(size):
        image = np.asarray(dataset[i]).reshape((32, 32, 3)) * 255
        cv2.imwrite(path + 'fake' + str(i) + '.png', image)