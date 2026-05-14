# Math:
import numpy as np

# Misc:
import os
import json
from integrator import VisualizeDataset, Euler, MachineLearning

class Simulation:
	def __init__(self):
		# ===== PARAMETERS =====
		# Simulation:
		with open("config/simulation.json") as f:
			simulation_parameters = json.load(f)

		self.dt = simulation_parameters["dt"]
		self.N_x = simulation_parameters["N_x"]
		self.N_y = simulation_parameters["N_y"]
		initialization_noise_std = simulation_parameters["initialization_noise_std"]
		fastened_particles = simulation_parameters["fastened_particles"]

		# Settings:
		with open("config/settings.json") as f:
			settings = json.load(f)

		self.resolution = settings["resolution"]
		self.create_dataset = settings["create_dataset"]
		use_ML = settings["use_ML"]
		visualize_dataset = settings["visualize_dataset"]
		dataset_path = settings["dataset_path"]
		# ======================

		# Initialize fabric:
		pos = self.init_body()
		self.mouse_down = False
		self.mouse_pos = np.zeros(3)
		self.chosen_particle = False
		self.closest_particle_distance = np.inf

		# Fastened_particles:
		fastened_mask = np.zeros(pos.shape, dtype = bool)
		if fastened_particles == "left":
			fastened_mask[:, :, 0] = True
		elif fastened_particles == "roof":
			fastened_mask[:, -1, :] = True
		elif fastened_particles == "floor":
			fastened_mask[:, 0, :] = True

		all_but_fastened = ~fastened_mask

		# Disturb fabric:
		noise = np.random.normal(0, initialization_noise_std, pos.shape)

		# Initialize engine:
		if self.create_dataset:
			if visualize_dataset:
				raise RuntimeError('Selected "visualize dataset" but also asked to generate a dataset')
			if use_ML:
				raise RuntimeError("Selected a Machine Learned engine model but also asked to generate a dataset")

		if visualize_dataset:
			self.engine = VisualizeDataset()
		elif use_ML:
			self.engine = MachineLearning(pos, noise, all_but_fastened)
		else:
			self.engine = Euler(pos, noise, all_but_fastened)

		# Initialize dataset saver:
		if self.create_dataset:
			self.pos_path = dataset_path + "pos"
			self.vel_path = dataset_path + "vel"
			self.forces_path = dataset_path + "forces"

			self.force_list = []
			self.pos_list = []
			self.vel_list = []
			
			npy_list = [int(f.split(".")[0]) for f in os.listdir(self.forces_path)]
			self.set_nr = max(npy_list)+1 if npy_list else 0

	def init_body(self):
		x = np.linspace(0, self.N_x-1, self.N_x)
		y = np.linspace(0, self.N_y-1, self.N_y)
		z = np.zeros(self.N_x)
		X, Y = np.meshgrid(x, y)
		Z = np.zeros_like(X)
		positions = np.stack((X, Y, Z), axis=0)
		return positions

	def get_transform(self):
		pos = self.engine.get_pos()

		min_x = pos[0].min()
		max_x = pos[0].max()
		min_y = pos[1].min()
		max_y = pos[1].max()

		self.x_scale = 0.75/(max_x-min_x)
		self.y_scale = 0.75/(max_y-min_y)

		self.x_offset = -(min_x+max_x)/2*self.x_scale
		self.y_offset = -(min_y+max_y)/2*self.y_scale

		return self.x_scale, self.y_scale, self.x_offset, self.y_offset

	def screen_to_sim(self, x, y):
		unscaled_x = 2*(x/self.resolution[0])-1
		unscaled_y = 2*((self.resolution[1]-y)/self.resolution[1])-1

		x_sim = (unscaled_x-self.x_offset) / self.x_scale
		y_sim = (unscaled_y-self.y_offset) / self.y_scale
		return np.array([x_sim, y_sim])

	def on_mouse_move(self, x, y):
		self.mouse_pos[:2] = self.screen_to_sim(x, y)

	def on_mouse_click(self, x, y):
		self.click_at = self.screen_to_sim(x, y)

		if not self.chosen_particle:
			distances_to_mouse = np.linalg.norm(self.engine.get_pos()[:2]-self.click_at[:, None, None], axis=0)
			flat_index = np.argmin(distances_to_mouse)
			self.chosen_particle = np.unravel_index(flat_index, distances_to_mouse.shape)
			self.closest_particle_distance = distances_to_mouse[self.chosen_particle]

			self.mouse_pos[2] = self.engine.get_pos()[2, *self.chosen_particle]

	def on_mouse_release(self):
		self.chosen_particle = False

	def step(self):
		if (self.chosen_particle is not False) and self.mouse_down:
			self.engine.move_point(self.chosen_particle, self.mouse_pos)

		if self.create_dataset:
			self.record_pos_vel()

		self.engine.step()

		if self.create_dataset:
			self.record_forces()

	@property
	def pos(self):
		return self.engine.get_pos()

	def record_pos_vel(self):
		if not self.mouse_down:
			self.pos_list.append(self.engine.get_pos())
			self.vel_list.append(self.engine.get_vel())
		elif len(self.force_list) > 0:
			self.save_pos_vel()
			self.pos_list = []
			self.vel_list = []

	def record_forces(self):
		if not self.mouse_down:
			self.force_list.append(self.engine.get_forces())
		elif len(self.force_list) > 0:
			self.save_forces()
			self.force_list = []
			self.set_nr += 1

	def save_pos_vel(self):
		np.save(f"{self.pos_path}/{self.set_nr}.npy", np.array(self.pos_list))
		np.save(f"{self.vel_path}/{self.set_nr}.npy", np.array(self.vel_list))

	def save_forces(self):
		np.save(f"{self.forces_path}/{self.set_nr}.npy", np.array(self.force_list))

	def end_sim(self):
		if self.create_dataset:
			self.save_pos_vel()
			self.save_forces()
			print("Saved dataset")