import os
from tqdm import tqdm
from PIL import Image
import warnings
import logger
import torch


def load_dataset(path, transforms):
    dataset = []
    for file in tqdm(os.listdir(path + "/")):
        # Load images and their class into features and targets
        image = Image.open(path + "/" + file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            image = image.convert('RGB')
        if image is not None:
            image = transforms(image)
            dataset.append(image)
    dataset = torch.concat(dataset).reshape((-1, 3, 256, 256))
    logger.start_log(f'Loaded {dataset.shape[0]} files from {path}')
    return dataset
