# Math:
import numpy as np

# NN:
import torch
import nn as my_models

# Misc:
import json
from chunk_gen import ChunkGen

class VisualizeDataset:
	def __init__(self):
		# ===== PARAMETERS =====
		# Simulation:
		with open("config/simulation.json") as f:
			simulation_parameters = json.load(f)

		self.dt = simulation_parameters["dt"]

		# Settings:
		with open("config/settings.json") as f:
			settings = json.load(f)

		temp_path = settings["temp_path"]
		# ======================

		self.pos_vel_path = temp_path + "/pos_vel_"

		chunks = ChunkGen()

		# First step:
		self.chunk_nr = 0
		self.step_counter = 0
		self.load_chunk()
		self.pos = self.positions[self.step_counter]

	def load_chunk(self):
		try:
			self.positions = np.load(self.pos_vel_path + f"{self.chunk_nr}.npy")[:, :3, :, :]
			self.chunk_size = self.positions.shape[0]
		except FileNotFoundError:
			print("Finished!")
		
	def step(self):
		if self.step_counter == self.chunk_size:
			self.chunk_nr += 1
			self.step_counter = 0
			self.load_chunk()

		self.pos = self.positions[self.step_counter]
		self.step_counter += 1

	def get_pos(self):
		return self.pos.copy()


class Euler:
	def __init__(self, pos, noise, all_but_fastened):
		# ===== PARAMETERS =====
		# Simulation:
		with open("config/simulation.json") as f:
			simulation_parameters = json.load(f)

		self.dampening_coefficient = simulation_parameters["dampening_coefficient"]
		self.k_structural = simulation_parameters["k_structural"]
		self.k_shear = simulation_parameters["k_shear"]
		self.k_bend = simulation_parameters["k_bend"]
		self.g = simulation_parameters["g"]
		self.mass = simulation_parameters["mass"]
		self.simulation_noise_std = simulation_parameters["simulation_noise_std"]
		self.auto_dragging_prob = simulation_parameters["auto_dragging_prob"]
		self.compression_prob = simulation_parameters["compression_prob"]
		self.dt = simulation_parameters["dt"]
		# ======================

		self.pos = pos
		self.vel = np.zeros_like(self.pos)
		self.total_forces = np.zeros_like(self.pos)
		self.all_but_fastened = all_but_fastened

		self.fastened_pos = self.pos[~self.all_but_fastened]

		self.relaxed_distances = self.neighbours(self.pos)
		self.relaxed_bending = self.neighbours_bending(self.pos)

		self.pos[self.all_but_fastened] += noise[self.all_but_fastened]

	def directions(self, distances):
		norms = np.linalg.norm(distances, axis=0, keepdims=True)
		np.clip(norms, 1e-6, 100, out=norms)
		np.clip(distances, -100, 100, out=distances)
		return distances/norms

	def structural_spring_force(self):
		spring_directions = self.directions(self.distances)
		relaxed_length = np.linalg.norm(self.relaxed_distances, axis = 0, keepdims = True)
		length = np.linalg.norm(self.distances, axis = 0, keepdims = True)

		spring_force_contributions = - self.k_structural * (length[..., :4] - relaxed_length[..., :4]) * spring_directions[..., :4]

		self.total_forces += spring_force_contributions.sum(axis=3)

	def shearing_spring_force(self):
		spring_directions = self.directions(self.distances)
		relaxed_length = np.linalg.norm(self.relaxed_distances, axis = 0, keepdims = True)
		length = np.linalg.norm(self.distances, axis = 0, keepdims = True)

		spring_force_contributions = - self.k_shear * (length[..., 4:] - relaxed_length[..., 4:]) * spring_directions[..., 4:]

		self.total_forces += spring_force_contributions.sum(axis=3)

	def bending_spring_force(self):
		relative_pos = self.neighbours_bending(self.pos)
		bending_direction = self.directions(relative_pos)

		relaxed_bending_magnitude = np.linalg.norm(self.relaxed_bending, axis = 0, keepdims = True)
		bending = np.linalg.norm(relative_pos, axis = 0, keepdims = True)

		spring_force_contributions = - self.k_bend * (bending - relaxed_bending_magnitude) * bending_direction

		self.total_forces += spring_force_contributions.sum(axis=3)

	def dampening_force(self):
		self.relative_vel = self.neighbours(self.vel)
		spring_directions = self.directions(self.distances)

		relative_velocities_along_spring = (self.relative_vel * spring_directions).sum(axis = 0, keepdims = True) * spring_directions

		dampening_forces = - self.dampening_coefficient * relative_velocities_along_spring.sum(axis = 3)
		self.total_forces += dampening_forces

	def gravity_force(self):
		self.total_forces[1] -= self.g * self.mass

	def neighbours(self, particles):
		differences = np.zeros((*particles.shape, 8))

		# Orthogonal:
		differences[:, 1:, :, 0] = particles[:, 1:, :] - particles[:, :-1, :] # W
		differences[:, :-1, :, 1] = particles[:, :-1, :] - particles[:, 1:, :] # E
		differences[:, :, 1:, 2] = particles[:, :, 1:] - particles[:, :, :-1] # N
		differences[:, :, :-1, 3] = particles[:, :, :-1] - particles[:, :, 1:] # S

		# Diagonal:
		differences[:, 1:, 1:, 4] = particles[:, 1:, 1:] - particles[:, :-1, :-1] # NW
		differences[:, :-1, 1:, 5] = particles[:, :-1, 1:] - particles[:, 1:, :-1] # NE
		differences[:, 1:, :-1, 6] = particles[:, 1:, :-1] - particles[:, :-1, 1:] # SW
		differences[:, :-1, :-1, 7] = particles[:, :-1, :-1] - particles[:, 1:, 1:] # SE

		np.nan_to_num(differences, copy=False, nan=0.1, posinf=100, neginf=-100)
		return np.clip(differences, -100, 100, out=differences)

	def neighbours_bending(self, particles):
		differences = np.zeros((*particles.shape, 4))

		# Orthogonal:
		differences[:, 2:, :, 0] = particles[:, 2:, :] - particles[:, :-2, :] # W
		differences[:, :-2, :, 1] = particles[:, :-2, :] - particles[:, 2:, :] # E
		differences[:, :, 2:, 2] = particles[:, :, 2:] - particles[:, :, :-2] # N
		differences[:, :, :-2, 3] = particles[:, :, :-2] - particles[:, :, 2:] # S

		np.nan_to_num(differences, copy=False, nan=0.1, posinf=100, neginf=-100)
		return np.clip(differences, -100, 100, out=differences)

	def move_point(self, particle_idx, pointer):
		self.pos[:, *particle_idx] = pointer
		self.vel[:, *particle_idx] = 0

	def step(self, allow_manipulation=False):
		self.total_forces = np.zeros_like(self.pos)
		self.distances = self.neighbours(self.pos)
		self.structural_spring_force()
		self.shearing_spring_force()
		self.bending_spring_force()
		self.dampening_force()
		self.gravity_force()
		np.nan_to_num(self.total_forces, copy=False, nan=0.1, posinf=100, neginf=-100)
		np.clip(self.total_forces, -100, 100, out=self.total_forces)

		self.acc = self.total_forces/self.mass

		self.vel[self.all_but_fastened] += self.acc[self.all_but_fastened]*self.dt
		np.nan_to_num(self.vel, copy=False, nan=0.1, posinf=100, neginf=-100)
		np.clip(self.vel, -100, 100, out=self.vel)
		self.pos[self.all_but_fastened] += self.vel[self.all_but_fastened]*self.dt
		np.nan_to_num(self.pos, copy=False, nan=0.1, posinf=100, neginf=-100)
		np.clip(self.pos, -100, 100, out=self.pos)

		# Manipulate fabric:
		if allow_manipulation:
			self.manipulate()

		# Reset fastened:
		self.pos[~self.all_but_fastened] = self.fastened_pos
		self.vel[~self.all_but_fastened] = 0

		if np.any(np.isnan(self.pos)) or np.any(np.isnan(self.vel)) or np.any(np.isnan(self.total_forces)):
			string = "Nan in"

			if np.any(np.isnan(self.pos)):
				string += " pos"

			if np.any(np.isnan(self.vel)):
				string += " vel"

			if np.any(np.isnan(self.total_forces)):
				string += " forces"
				
			raise ValueError(string)

	def get_pos(self):
		return self.pos.copy()

	def get_forces(self):
		return self.total_forces.copy()

	def get_vel(self):
		return self.vel.copy()

	# Fabric manipulation:
	def manipulate(self):
		if self.simulation_noise_std:
			"""
			Noise is added to the output position of each step
			"""
			self.pos[self.all_but_fastened] += np.clip(np.random.normal(
							0,
							self.simulation_noise_std,
							self.pos.shape
							)[self.all_but_fastened], -1, 1)
			self.vel[self.all_but_fastened] += np.clip(np.random.normal(
							0,
							self.simulation_noise_std,
							self.vel.shape
							)[self.all_but_fastened], -1, 1)
		
		if self.auto_dragging_prob:
			"""
			Points are dragged away randomly
			"""
			r = np.random.random()
			if r < self.auto_dragging_prob:
				x, y = np.random.randint(0, min(self.pos[0, :, :].shape), size=2)
				self.pos[:, x, y] += np.random.uniform(-10, 10, 3)
				self.vel[:, x, y] = 0
		
		if self.compression_prob:
			"""
			Fabric is compressed into ball and then relaxed
			"""
			r = np.random.random()
			if r < self.compression_prob:
				center_of_compression = np.zeros((3, 1, 1))
				center_of_compression[0] = self.pos.shape[1]/2
				center_of_compression[1] = self.pos.shape[2]
				self.pos = center_of_compression + 0.2*(self.pos - center_of_compression)
				self.vel[:] = 0

class MachineLearning:
	def __init__(self, pos, noise, all_but_fastened):
		# ===== PARAMETERS =====
		# Simulation:
		with open("config/simulation.json") as f:
			simulation_parameters = json.load(f)

		self.mass = simulation_parameters["mass"]
		self.dt = simulation_parameters["dt"]
		self.max_vel = simulation_parameters["max_vel"]

		# Model:
		with open("config/model.json") as f:
			model_parameters = json.load(f)

		input_channels = model_parameters["input_channels"]
		output_channels = model_parameters["output_channels"]
		hidden_channels = model_parameters["hidden_channels"]
		dilations = model_parameters["dilations"]
		padding_mode = model_parameters["padding_mode"]
		use_residual_blocks = model_parameters["use_residual_blocks"]

		# Settings:
		with open("config/settings.json") as f:
			settings = json.load(f)

		model_path = settings["model_path"]
		compile_model = settings["compile_model"]
		# ======================	

		# Build model:
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		print("Device: " f"{self.device}")

		self.model = my_models.CNN(
			input_channels,
	        output_channels,
	        hidden_channels=hidden_channels,
	        use_residual_blocks=use_residual_blocks,
	        dilations=dilations,
	        padding_mode=padding_mode
			).to(self.device)
		state = torch.load(model_path + "model.pth", map_location=torch.device(self.device), weights_only=True)
		self.model.load_state_dict(state)
		self.model.eval()
		if compile_model:
			self.model = torch.compile(self.model)

		# Parameter count:
		print("Trainable parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

		self.pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
		self.vel = torch.zeros_like(self.pos)
		self.all_but_fastened = torch.tensor(all_but_fastened, dtype=torch.bool, device=self.device)

		noise = torch.tensor(noise, dtype=torch.float32, device=self.device)
		self.pos[self.all_but_fastened] += noise[self.all_but_fastened]

		# Load normalisation constant:
		with open("config/norm.json", "r") as f:
			norm = json.load(f)

		self.force_scale = norm["force_scale"]

	def move_point(self, particle_idx, pointer):
		if not isinstance(self.pos, np.ndarray):
			pointer = torch.tensor(pointer, dtype=torch.float32, device=self.device)
		self.pos[:, *particle_idx] = pointer
		self.vel[:, *particle_idx] = 0

	def step(self):
		with torch.no_grad():
			self.total_forces = my_models.run(self.model, torch.cat((self.pos, self.vel), axis=0).unsqueeze(0), self.device)
			self.total_forces = self.total_forces.squeeze(0)
			self.total_forces *= self.force_scale

		self.acc = self.total_forces/self.mass
		self.vel[self.all_but_fastened] += self.acc[self.all_but_fastened]*self.dt
		self.vel.clamp_(-self.max_vel, self.max_vel)
		self.pos[self.all_but_fastened] += self.vel[self.all_but_fastened]*self.dt

	def get_pos(self):
		return self.pos.cpu().numpy()