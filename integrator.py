# Math:
import numpy as np

# NN:
import torch
import nn as my_models

# Misc:
import json

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

		dataset_path = settings["dataset_path"]
		pos_path = dataset_path + "pos"
		vel_path = dataset_path + "vel"
		forces_path = dataset_path + "forces"
		# ======================

		i = 0
		self.positions = []
		self.velocities = []
		self.forces = []
		while True:
			try:
				temp_pos = np.load(pos_path + f"/{i}.npy")
				temp_vel = np.load(vel_path + f"/{i}.npy")
				temp_forces = np.load(forces_path + f"/{i}.npy")
				self.positions.append(temp_pos.copy())
				self.velocities.append(temp_vel.copy())
				self.forces.append(temp_forces.copy())
				i += 1
			except FileNotFoundError:
				break

		self.positions = np.concatenate(self.positions, axis=0)
		self.velocities = np.concatenate(self.velocities, axis=0)
		self.forces = np.concatenate(self.forces, axis=0)

		self.step_counter = 0

		# First step:
		self.pos = self.positions[0]
		
	def step(self):
		try:
			self.step_counter += 1
			self.pos = self.positions[self.step_counter]
		except IndexError:
			pass

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
		self.dt = simulation_parameters["dt"]
		# ======================

		self.pos = pos
		self.vel = np.zeros_like(self.pos)
		self.total_forces = np.zeros_like(self.pos)
		self.all_but_fastened = all_but_fastened

		self.relaxed_distances = self.neighbours(self.pos)
		self.relaxed_bending = self.neighbours_bending(self.pos)

		self.pos[self.all_but_fastened] += noise[self.all_but_fastened]

	def structural_spring_force(self):
		spring_directions = self.distances / (np.linalg.norm(self.distances, axis=0, keepdims=True) + 1e-12)
		relaxed_length = np.linalg.norm(self.relaxed_distances, axis = 0, keepdims = True)
		length = np.linalg.norm(self.distances, axis = 0, keepdims = True)

		spring_force_contributions = - self.k_structural * (length[..., :4] - relaxed_length[..., :4]) * spring_directions[..., :4]

		self.total_forces += spring_force_contributions.sum(axis=3)

	def shearing_spring_force(self):
		spring_directions = self.distances / (np.linalg.norm(self.distances, axis=0, keepdims=True) + 1e-12)
		relaxed_length = np.linalg.norm(self.relaxed_distances, axis = 0, keepdims = True)
		length = np.linalg.norm(self.distances, axis = 0, keepdims = True)

		spring_force_contributions = - self.k_shear * (length[..., 4:] - relaxed_length[..., 4:]) * spring_directions[..., 4:]

		self.total_forces += spring_force_contributions.sum(axis=3)

	def bending_spring_force(self):
		relative_pos = self.neighbours_bending(self.pos)
		bending_direction = relative_pos / (np.linalg.norm(relative_pos, axis=0, keepdims=True) + 1e-12)

		relaxed_bending_magnitude = np.linalg.norm(self.relaxed_bending, axis = 0, keepdims = True)
		bending = np.linalg.norm(relative_pos, axis = 0, keepdims = True)

		spring_force_contributions = - self.k_bend * (bending - relaxed_bending_magnitude) * bending_direction

		self.total_forces += spring_force_contributions.sum(axis=3)

	def dampening_force(self):
		self.relative_vel = self.neighbours(self.vel)
		spring_directions = self.distances / (np.linalg.norm(self.distances, axis=0, keepdims=True) + 1e-12)

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

		return differences

	def neighbours_bending(self, particles):
		differences = np.zeros((*particles.shape, 4))

		# Orthogonal:
		differences[:, 2:, :, 0] = particles[:, 2:, :] - particles[:, :-2, :] # W
		differences[:, :-2, :, 1] = particles[:, :-2, :] - particles[:, 2:, :] # E
		differences[:, :, 2:, 2] = particles[:, :, 2:] - particles[:, :, :-2] # N
		differences[:, :, :-2, 3] = particles[:, :, :-2] - particles[:, :, 2:] # S

		return differences

	def move_point(self, particle_idx, pointer):
		self.pos[:, *particle_idx] = pointer
		self.vel[:, *particle_idx] = 0

	def step(self):
		self.total_forces = np.zeros_like(self.pos)
		self.distances = self.neighbours(self.pos)
		self.structural_spring_force()
		self.shearing_spring_force()
		self.bending_spring_force()
		self.dampening_force()
		self.gravity_force()

		self.acc = self.total_forces/self.mass

		self.vel[self.all_but_fastened] += self.acc[self.all_but_fastened]*self.dt
		self.pos[self.all_but_fastened] += self.vel[self.all_but_fastened]*self.dt

		# Add noise:
		self.pos[self.all_but_fastened] += np.random.normal(0, self.simulation_noise_std, self.pos.shape)[self.all_but_fastened]

	def get_pos(self):
		return self.pos.copy()

	def get_forces(self):
		return self.total_forces.copy()

	def get_vel(self):
		return self.vel.copy()


class MachineLearning:
	def __init__(self, pos, noise, all_but_fastened):
		# ===== PARAMETERS =====
		# Simulation:
		with open("config/simulation.json") as f:
			simulation_parameters = json.load(f)

		self.mass = simulation_parameters["mass"]
		self.dt = simulation_parameters["dt"]

		# Model:
		with open("config/model.json") as f:
			model_parameters = json.load(f)

		network = model_parameters["network"]
		network_args = model_parameters["network_args"]

		# Settings:
		with open("config/settings.json") as f:
			settings = json.load(f)

		model_path = settings["model_path"]
		compile_model = settings["compile_model"]
		# ======================	

		# Architectures:
		self.architecture = {
		    "CNN": my_models.CNN
		    }

		# Build model:
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		print("Device: " f"{self.device}")

		self.model = self.architecture[network](*network_args).to(self.device)
		state = torch.load(model_path, map_location=torch.device(self.device), weights_only=True)
		self.model.load_state_dict(state)
		self.model.eval()
		if compile_model:
			self.model = torch.compile(self.model)

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

	def step(self):
		with torch.no_grad():
			self.total_forces = my_models.run(self.model, torch.cat((self.pos, self.vel), axis=0), self.device)
			self.total_forces *= self.force_scale
			# self.total_forces = torch.clamp(self.total_forces, -self.force_scale, self.force_scale)

		self.acc = self.total_forces/self.mass
		self.vel[self.all_but_fastened] += self.acc[self.all_but_fastened]*self.dt
		self.pos[self.all_but_fastened] += self.vel[self.all_but_fastened]*self.dt

	def get_pos(self):
		return self.pos.cpu().numpy()