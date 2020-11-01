import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
from os.path import join
import torch
import numpy as np
from collections import defaultdict
transform = transforms.Compose([
	# transforms.Resize((224, 224)),
	transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1)),
])

class DataloaderSegmentation(Dataset):
	def __init__(self, root='./hw2_data/p2_data', mode='train'):
		assert mode == 'train' or mode == 'validation', 'need to be train or val'
		self.root = join(root, mode)
		self.file_length = len(os.listdir(self.root)) // 2
		self.filenames = ["{:>04}".format(i) for i in range(self.file_length)]
		self.name_to_color = {'Urban' : [0, 255, 255], 'Agriculture' : [255, 255, 0], 'Rangeland' : [255, 0, 255], 'Forest' : [0, 255, 0], 'Water' : [0, 0, 255], 'Barren' : [255, 255, 255], 'Unknown' : [0, 0, 0]}
		self.name_to_class = {'Urban' : 0, 'Agriculture' : 1, 'Rangeland' : 2, 'Forest' : 3, 'Water' : 4, 'Barren' : 5, 'Unknown' : 6}
		self.class_to_name = { 0 : 'Urban', 1: 'Agriculture', 2:'Rangeland', 3 : 'Forest', 4 : 'Water', 5 : 'Barren',  6 : 'Unknown'}
		# self.flip = transforms.RandomFlip(1)
	def __len__(self):
		return self.file_length

	def __getitem__(self, idx):
		names = self.filenames[idx]
		image_name = names + "_sat.jpg"
		label_name = names + "_mask.png"
		image = Image.open(join(self.root, image_name))
		image = transform(image)
		label_rgb = np.array(Image.open(join(self.root, label_name)))
		label = np.ones(label_rgb.shape[:2]) * 6
		for l, n in self.name_to_class.items():
			x, y = np.where(np.all(label_rgb == self.name_to_color[l], axis=-1))
			label[x, y] = n
		label = torch.LongTensor(label)
		return {'image' : image, 'label' : label, 'filename' : names}

# x = DataloaderSegmentation()
# import matplotlib.pyplot as plt
# q = x[0]
# a = q['image']
# b = q['label']

# a = a.permute(1, 2, 0)
# a = a.numpy()
# b = b.numpy()
# plt.imshow(b)
# plt.show()