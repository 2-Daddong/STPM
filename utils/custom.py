import os
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class CustomDataset(Dataset):
	def __init__(self, dataset_path='./STPM/datasets/', class_name = 'cavity', is_train=True, 
					inputsize=(256,256), resize=(500,500), cropsize=(190,700)):

		self.dataset_path = dataset_path
		self.class_name = class_name
		self.is_train = is_train
		self.inputsize = inputsize
		self.resize = resize
		self.cropsize = cropsize
		gap = self.cropsize[0] - self.cropsize[1]

		if gap < 0:
			self.pad = (0, int(abs(gap)/2))
		else:
			self.pad = (int(abs(gap)/2), 0)

		# load dataset
		self.x, self.y = self.load_dataset_folder()

		self.transform_x = T.Compose([#T.Resize(self.resize, Image.ANTIALIAS),
										#T.CenterCrop(self.cropsize),
										#T.Pad(self.pad),
										T.Resize(self.inputsize, Image.ANTIALIAS),
										T.ToTensor(),
										T.Normalize(mean=[0.485, 0.456, 0.406],
													std=[0.229, 0.224, 0.225])])


	def __getitem__(self, idx):
		x, y = self.x[idx], self.y[idx]

		x = Image.open(x).convert('RGB')
		x = self.transform_x(x)

		return x, y


	def __len__(self):
		return len(self.x)


	def load_dataset_folder(self):
		phase = 'train' if self.is_train else 'test'
		x, y = [], []

		img_dir = os.path.join(self.dataset_path, self.class_name, phase)
		img_types = sorted(os.listdir(img_dir))

		for img_type in img_types:

			# load images
			img_type_dir = os.path.join(img_dir, img_type)
			if not os.path.isdir(img_type_dir):
				continue

			img_fpath_list = sorted([os.path.join(img_type_dir, f)
									for f in os.listdir(img_type_dir)])
			x.extend(img_fpath_list)

			# load gt labels
			if img_type == 'OK':
				y.extend([0] * len(img_fpath_list)) #OK label=0
			else:
				y.extend([1] * len(img_fpath_list)) #NG label=1


		assert len(x) == len(y), 'number of x and y should be same'

		return list(x), list(y)