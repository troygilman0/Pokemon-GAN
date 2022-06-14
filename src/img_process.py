import os
from tqdm import tqdm
from PIL import Image
import warnings
import logger


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
            dataset.append((image, 0))
    logger.start_log(f'Loaded {len(dataset)} files from {path}')
    return dataset
