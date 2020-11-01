import torch
import torch.nn as nn





class DiceCoefficientLoss(nn.Module):

	def __init__(self):

		super().__init__()

	def forward(self, pred, label):

		b, _ = pred.size()
		pred = pred.view(b, 512, 512, 7)
		pred = pred.argmax(-1)

		label = label.view(b, 512, 512)

		
