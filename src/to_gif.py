from PIL import Image
from tqdm import tqdm
import os

dir_path = 'out/2022-08-04 09:44:27.458229/fake_data/'
gif_path = 'pokegan.gif'
duration = 50
img_size = (2048, 308)

file_dict = {}
print(f'Loading files from {dir_path}...')
for filename in tqdm(os.listdir(dir_path)):
    img = Image.open(dir_path + filename)
    num = int(filename[5:-4])
    file_dict[num] = img

print('Sorting files...')
imgs = tqdm([file_dict[num] for num in sorted(file_dict)])

print(f'Resizing files to {img_size}...')
#imgs = tqdm([img.resize(img_size, resample=Image.BOX) for img in imgs])

imgs = (img for img in imgs)
iter = next(imgs)

print(f'Saving GIF to {gif_path}...')
iter.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, duration=duration, loop=0)
print('Finished')