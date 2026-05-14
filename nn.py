# NN:
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
	def __init__(self, input_channels, output_channels, hidden_channels=[], kernel_size=3):
		super().__init__()

		# Channels:
		channels = [input_channels] + hidden_channels + [output_channels]

		# Convolutional layers:
		self.conv = nn.ModuleList()
		for i in range(len(channels)-1):
			self.conv.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=kernel_size, padding=int((kernel_size-1)/2)))

	def forward(self, x):
		for i, layer in enumerate(self.conv):
			x = layer(x)
			if i != (len(self.conv)-1):
				x = activation(x)
		return x

def activation(x):
	return F.relu(x)

def train(model, loader, optimizer, criterion, device):
	total_loss = 0
	model.train()
	for pos, targets in loader:
		optimizer.zero_grad()
		logits = model(pos)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
	return total_loss

def evaluate(model, loader, criterion, device):
	total_loss = 0
	predict_pos = []
	true_pos = []

	model.eval()
	with torch.no_grad():
		for pos, targets in loader:
			logits = model(pos)
			loss = criterion(logits, targets)
			total_loss += loss.item()

	return total_loss

def run(model, pos_vel, device):
	logits = model(pos_vel)
	predict_forces = logits
	return predict_forces