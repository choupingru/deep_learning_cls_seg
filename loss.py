import torch
import torch.nn as nn



class DiceLoss(nn.Module):

	def forward(self, pred, label):

		intersection = pred * label
		union = pred + label
		
		return 1 - (2 * intersection.sum()) / (union + 1e-5)


class DiceCoefficientLoss(nn.Module):

	def __init__(self, num_classes=7):
		super().__init__()
		self.num_classes = num_classes
		self.dice_loss = DiceLoss()
	def forward(self, pred, label):

		b, _ = pred.size()
		pred = pred.view(b, 512, 512, 7)
		label = label.view(b, 512, 512)

		loss = 0
		for i in range(self.num_classes):
			p = pred[..., i]
			l = label * (label == i) / i
			loss += self.dice_loss(p, l)

		return loss / self.num_classes

