# NN:
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
	def __init__(
		self,
		input_channels,
		output_channels,
		hidden_channels=[],
		kernel_size=3,
		use_residual_blocks=False,
		dilations=False,
		padding_mode="zeros"
		):
		super().__init__()

		# Channels:
		channels = [input_channels] + hidden_channels + [output_channels]

		# Convolutional layers:
		self.layers = nn.ModuleList()

		if dilations is False:
			dilations = [1 for _ in range(len(channels)-1)]

		for i in range(len(channels)-1):
			in_channels = channels[i]
			out_channels = channels[i+1]
			dilation = dilations[i]
			last_layer = True if i == len(channels)-2 else False
			padding = dilation*(kernel_size//2)

			if use_residual_blocks and in_channels == out_channels:
				self.layers.append(ResidualBlock(
					in_channels,
					kernel_size=kernel_size,
					dilation=dilation,
					padding=padding,
					padding_mode=padding_mode
					))

			else:
				self.layers.append(ConvBlock(
					in_channels,
					out_channels,
					kernel_size=kernel_size,
					dilation=dilation,
					padding=padding,
					last_layer=last_layer,
					padding_mode=padding_mode
					))

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x


class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1, last_layer=False, padding_mode="zeros"):
		super().__init__()
		self.last_layer = last_layer
		
		self.conv = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=kernel_size,
			padding=padding,
			dilation=dilation,
			padding_mode=padding_mode
			)

		self.batch_norm = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = self.conv(x)
		x = self.batch_norm(x)
		if not self.last_layer:
			x = activation(x)
		return x


class ResidualBlock(nn.Module):
	def __init__(self, channels, kernel_size=3, dilation=1, padding=1, padding_mode="zeros"):
		super().__init__()
		self.conv1 = nn.Conv2d(
			channels,
			channels,
			kernel_size=kernel_size,
			padding=padding,
			dilation=dilation,
			padding_mode=padding_mode
			)
		self.batch_norm_1 = nn.BatchNorm2d(channels)

		self.conv2 = nn.Conv2d(
			channels,
			channels,
			kernel_size=kernel_size,
			padding=padding,
			dilation=dilation,
			padding_mode=padding_mode
			)
		self.batch_norm_2 = nn.BatchNorm2d(channels)

	def forward(self, x):
		y = self.conv1(x)
		y = self.batch_norm_1(y)
		y = activation(y)

		y = self.conv2(y)
		y = self.batch_norm_2(y)
		return x + y


def activation(x):
	return F.gelu(x)

def train(model, loader, optimizer, criterion, device):
	total_loss = 0
	model.train()
	for pos_vel, forces in loader:
		pos_vel = pos_vel.to(device)
		forces = forces.to(device)

		optimizer.zero_grad()
		logits = model(pos_vel)
		loss = criterion(logits, forces)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	return total_loss

def evaluate(model, loader, criterion, device):
	total_loss = 0
	model.eval()
	with torch.no_grad():
		for pos_vel, forces in loader:
			pos_vel = pos_vel.to(device)
			forces = forces.to(device)

			logits = model(pos_vel)
			loss = criterion(logits, forces)
			total_loss += loss.item()
	return total_loss

def run(model, pos_vel, device):
	logits = model(pos_vel)
	predict_forces = logits
	return predict_forces