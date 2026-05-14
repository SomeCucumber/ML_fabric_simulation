# Math:
import numpy as np

# NN:
import torch

# Misc:
import json

class Loader(torch.utils.data.Dataset):
	def __init__(self, N_x, N_y, dt, device):
		# ===== PARAMETERS =====
		with open("config/settings.json") as f:
			settings = json.load(f)

		dataset_path = settings["dataset_path"]
		pos_path = dataset_path + "pos"
		vel_path = dataset_path + "vel"
		forces_path = dataset_path + "forces"
		# ======================

		i = 0
		self.pos = []
		self.vel = []
		self.forces = []
		while True:
			try:
				temp_pos = np.load(pos_path + f"/{i}.npy")
				temp_vel = np.load(vel_path + f"/{i}.npy")
				temp_forces = np.load(forces_path + f"/{i}.npy")
				self.pos.append(temp_pos.copy())
				self.vel.append(temp_vel.copy())
				self.forces.append(temp_forces.copy())
				i += 1
			except FileNotFoundError:
				break

		self.pos = np.concatenate(self.pos, axis=0)
		self.vel = np.concatenate(self.vel, axis=0)
		self.forces = np.concatenate(self.forces, axis=0)
		self.pos_vel = np.concatenate((self.pos, self.vel), axis=1)

		# Normalisation:
		pos_scale = max(N_x, N_y)
		vel_scale = 1.0
		force_scale = 50.0

		self.pos_vel[:3] /= pos_scale
		self.pos_vel[3:] /= vel_scale
		self.forces /= force_scale

		# Save normalisation:
		normalistaion_constants = {
			"pos_scale": pos_scale,
			"vel_scale": vel_scale,
			"force_scale": force_scale
			}
		json.dump(normalistaion_constants, open("config/norm.json", "w"))

		# Clipping:
		self.forces = np.clip(self.forces, -1.0, 1.0)

		self.pos_vel = torch.tensor(self.pos_vel, dtype=torch.float32, device=device)
		self.forces = torch.tensor(self.forces, dtype=torch.float32, device=device)

	def __len__(self):
		return self.pos.shape[0]

	def __getitem__(self, idx):
		return self.pos_vel[idx], self.forces[idx]