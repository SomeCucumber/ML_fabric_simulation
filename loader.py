# Math:
import numpy as np

# NN:
import torch

# Misc:
import json

class Loader(torch.utils.data.Dataset):
	def __init__(self, chunk_nr):
		# ===== PARAMETERS =====
		# Settings:
		with open("config/settings.json") as f:
			settings = json.load(f)

		temp_path = settings["temp_path"]
		self.chunk_size = settings["chunk_size"]

		# Simulation:
		with open("config/simulation.json") as f:
			simulation_parameters = json.load(f)

		N_x = simulation_parameters["N_x"]
		N_y = simulation_parameters["N_y"]
		# ======================

		pos_vel_path = temp_path + "/pos_vel_"
		forces_path = temp_path + "/forces_"

		# Load chunk:
		self.pos_vel = np.load(pos_vel_path + f"{chunk_nr}.npy")
		self.forces = np.load(forces_path + f"{chunk_nr}.npy")

		# Normalisation:
		pos_scale = max(N_x, N_y)
		vel_scale = 1.0
		force_scale = 50.0

		self.pos_vel[:, :3, ...] /= pos_scale
		self.pos_vel[:, 3:, ...] /= vel_scale
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

		self.pos_vel = torch.tensor(self.pos_vel, dtype=torch.float32)
		self.forces = torch.tensor(self.forces, dtype=torch.float32)

	def __len__(self):
		return self.forces.shape[0]

	def __getitem__(self, idx):
		return self.pos_vel[idx], self.forces[idx]