import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
from os.path import join
import torch
import numpy as np

transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.RandomAffine(10, translate=(0, 0.1), scale=(0.9, 1.1), shear=(1, 2), fillcolor=0),
	transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(0.4)
])
class DataloaderClassification(Dataset):
	def __init__(self, root='./hw2_data/p1_data/train_50', mode='train'):
		assert mode == 'train' or mode == 'val', 'need to be train or val'
		self.root = root.replace('train', mode)
		filenames = os.listdir(self.root)
		self.names_label = [(name, int(name.split('_')[0])) for name in filenames]
	def __len__(self):
		return len(self.names_label)
	def __getitem__(self, idx):
		name, label = self.names_label[idx]
		image = Image.open(join(self.root, name))
		image = transform(image)
		label = torch.LongTensor([label])
		return {'image' : image, 'label' : label}
# x = DataloaderClassification()