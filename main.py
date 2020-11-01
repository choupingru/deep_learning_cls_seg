from dataloader_cls import DataloaderClassification
from dataloader_seg import DataloaderSegmentation
import numpy as np
import torch
import torch.nn as nn
import os 
from tqdm import tqdm
from importlib import import_module
from torch.utils.data import DataLoader
import argparse
import time
from os.path import join
from PIL import Image
from mean_iou_evaluate import *
from loss import DiceCoefficientLoss
parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
					help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
					help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
					help='number of total epochs to run')

parser.add_argument('--b', '--batch-size', default=16, type=int,
					metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default="./results", type=str, metavar='SAVE',
					help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
					help='1 do test evaluation, 0 not')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
					help='save frequency')
parser.add_argument('--task', default=1, type=int, help='1 for cls, 2 for seg')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_loss = 1000000

def main():

	global best_loss

	if not os.path.isdir(join(args.save_dir, args.model)):
		os.mkdir(join(args.save_dir, args.model))

	model_root = 'net'
	model = import_module('{}.{}'.format(model_root, args.model))

	net = model.get_model()
	
	if args.resume:
		checkpoint = torch.load(args.resume)
		net.load_state_dict(checkpoint['state_dict'])
		best_loss = checkpoint['best_loss']
		start_epoch = checkpoint['epoch'] + 1
	else:
		start_epoch = 1

	if args.task == 1:
		train_loader = DataloaderClassification(mode='train')
		train_loader = DataLoader(train_loader, batch_size=args.b, shuffle=True, num_workers=4)
		val_loader = DataloaderClassification(mode='val')
		val_loader = DataLoader(val_loader, batch_size=args.b)
	else:
		train_loader = DataloaderSegmentation(mode='train')
		train_loader = DataLoader(train_loader, batch_size=args.b, shuffle=True, num_workers=4)
		val_loader = DataloaderSegmentation(mode='validation')
		val_loader = DataLoader(val_loader, batch_size=args.b)
	
	optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
	
	pytorch_total_params = sum(p.numel() for p in net.parameters())
	print("Total number of params = ", pytorch_total_params)

	if args.task == 1:
		criterion = torch.nn.CrossEntropyLoss()	
	else:
		criterion = DiceCoefficientLoss(7)
		# criterion = CrossEntropyLoss()
		
	criterion = criterion.to(device)
	net = net.to(device)
	for epoch in range(start_epoch, args.epochs + 1):
		# Train for one epoch
		print()
		train(train_loader, net, criterion, epoch, optimizer, args.task)
		# Evaluate on validation set
		val_loss = validation(val_loader, net, criterion, epoch, args.task)
		print()
		# Remember the best val_loss and save checkpoint
		is_best = val_loss < best_loss
		best_loss = min(val_loss, best_loss)

		if epoch % args.save_freq == 0 or is_best:
			state_dict = net.state_dict()
			state_dict = {k:v.cpu() for k, v in state_dict.items()}
			state = {'epoch': epoch,
					 'save_dir': args.save_dir,
					 'state_dict': state_dict,
					 'args': args,
					 'best_loss': best_loss}
			torch.save(state, os.path.join(args.save_dir, args.model, '{:>03d}.ckpt'.format(epoch)))
		

def train(data_loader, net, criterion, epoch, optimizer, task=1):
	 
	start_time = time.time()

	net.train()

	pbar = tqdm(data_loader, ncols=50)

	preds, labels = [], []
	total_loss = 0

	for i, datas in enumerate(pbar):

		image = datas['image'].to(device)
		label = datas['label'].to(device)

		pred = net(image)
		label = label.view(-1)
		loss = criterion(pred, label)	

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		total_loss += loss.item()

		pred = pred.argmax(1).view(-1).cpu().detach().numpy()
		label = label.cpu().detach().numpy()
		preds.append(pred)
		labels.append(label)

	end_time = time.time()

	preds = np.concatenate([p for p in preds], 0)
	labels = np.concatenate([l for l in labels], 0)
	
	if args.task == 1:
		tp = (preds == labels).sum()
		ac = tp / len(preds)
		print('Train Epoch : %d, AC : %3.2f, Loss : %3.2f, times : %3.2f' % (epoch, ac, total_loss, end_time - start_time))
	else:
		print('Train Epoch : %d, Loss : %3.f, times : %3.2f' % (epoch, total_loss, end_time - start_time))


def validation(data_loader, net, criterion, epoch, task=1):

	start_time = time.time()

	net.eval()

	pbar = tqdm(data_loader, ncols=50)

	preds, labels = [], []
	total_loss = 0
	if task == 2:
		cls2name = data_loader.dataset.class_to_name
		name2color = data_loader.dataset.name_to_color
	with torch.no_grad():

		for i, datas in enumerate(pbar):

			image = datas['image'].to(device)
			label = datas['label'].to(device)
			b = label.size(0)

			if task == 2:
				names = datas['filename']

			pred = net(image)
			label = label.view(-1)
			loss = criterion(pred, label)
			total_loss += loss.item()
			pred = pred.argmax(1).view(-1).cpu().detach().numpy()
			label = label.cpu().detach().numpy()
			preds.append(pred)
			labels.append(label)
			if task == 2:
				pred = pred.reshape(b, 512, 512)
				output = np.zeros((b, 512, 512, 3))
				for num_cls in range(7):
					b, x, y = np.where(pred == num_cls)
					output[b, x, y] = name2color[cls2name[num_cls]]
				for index, name in enumerate(names):
					out = output[index]
					out = Image.fromarray(out.astype(np.uint8))
					out.save('./pred_mask/'+name+'_pred.png')

	end_time = time.time()
	preds = np.concatenate([p for p in preds], 0)
	labels = np.concatenate([l for l in labels], 0)	

	if args.task == 1:
		tp = (preds == labels).sum()
		ac = tp / len(preds)
		print('Valid Epoch : %d, AC : %3.2f, Loss : %3.2f, times : %3.2f' % (epoch, ac, total_loss, end_time - start_time))
	else:
		pred = read_masks('./pred_mask')
		label = read_masks('./hw2_data/p2_data/validation')
		mean_iou_score(pred, label)
		print('Valid Epoch : %d, Loss : %3.2f, times : %3.2f' % (epoch, total_loss, end_time - start_time))
	
	return total_loss / len(data_loader)



if __name__ == '__main__':
	main()