# Math:
import numpy as np

# Misc:
import os
import json
from integrator import VisualizeDataset, Euler, MachineLearning

try:
	from iterator_print import percentage_print
except ModuleNotFoundError:
	percentage_print = None

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
		dataset_path = settings["dataset_path"][0]
		self.chunk_size = settings["chunk_size"]
		self.number_of_steps = settings["number_of_steps"]
		self.save_interval = settings["save_interval"]
		# ======================

		self.step_counter = 0
		self.percentage_print = percentage_print

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
			self.dataset_path = dataset_path

			self.force_list = []
			self.pos_list = []
			self.vel_list = []
			
			os.makedirs(self.dataset_path, exist_ok=True)
			npy_list = [int(f.split(".")[0]) for f in os.listdir(self.dataset_path)]
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

		self.x_scale = 0.5/(max_x-min_x)
		self.y_scale = 0.5/(max_y-min_y)

		self.x_offset = -(min_x+max_x)/2*self.x_scale
		self.y_offset = -(min_y+max_y)/2*self.y_scale
		self.y_offset = 0.45

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
		if (not self.number_of_steps) or (self.step_counter < self.number_of_steps):
			if (self.chosen_particle is not False) and self.mouse_down:
				self.engine.move_point(self.chosen_particle, self.mouse_pos)

			if self.create_dataset:
				if self.step_counter % self.save_interval == 0:
					self.record_pos_vel()

			if self.save_interval and self.create_dataset:
				allow_manipulation = True if (self.step_counter % self.save_interval == 0) else False
				self.engine.step(allow_manipulation=allow_manipulation)
			else:
				self.engine.step()

			if self.create_dataset:
				if self.step_counter % self.save_interval == 0:
					self.record_forces()

				if len(self.force_list) == self.chunk_size:
					self.save_data()

				self.step_counter += 1

				if self.number_of_steps and self.percentage_print:
					percentage_print(self.step_counter, self.number_of_steps+1, message="Steps completed")

	@property
	def pos(self):
		return self.engine.get_pos()

	# ===== DATASET CREATION =====
	def record_pos_vel(self):
		"""
		This function records all positions and velocities IF
		mousebutton is not down, if mouse click is detected record
		is saved to file, if record gets longer than chunk_size, it also
		gets saved to file.
		"""
		self.pos_list.append(self.engine.get_pos())
		self.vel_list.append(self.engine.get_vel())

	def record_forces(self):
		"""
		This function records all forces IF mousebutton is not down,
		if mouse click is detected record is saved to file, if
		record gets longer than chunk_size, it also gets saved to file.
		"""
		self.force_list.append(self.engine.get_forces())

	def save_data(self):
		data_chunk = np.concatenate((np.array(self.pos_list), np.array(self.vel_list), np.array(self.force_list)), axis=1)
		np.save(f"{self.dataset_path}/{self.set_nr}.npy", data_chunk)
		self.pos_list = []
		self.vel_list = []
		self.force_list = []
		print("\nSaved dataset")
		self.set_nr += 1
			
	def end_sim(self):
		if self.create_dataset:
			self.save_data()