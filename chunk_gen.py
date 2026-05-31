# Math:
import numpy as np

# Misc:
import os
import json
import shutil

class ChunkGen():
	def __init__(self):
		# ===== PARAMETERS =====
		# Settings:
		with open("config/settings.json") as f:
			settings = json.load(f)

		dataset_paths = settings["dataset_path"]
		self.temp_path = settings["temp_path"]
		self.chunk_size = settings["chunk_size"]

		# Simulation:
		with open("config/simulation.json") as f:
			simulation_parameters = json.load(f)

		N_x = simulation_parameters["N_x"]
		N_y = simulation_parameters["N_y"]
		# ======================

		# Empty temp folder:
		shutil.rmtree(self.temp_path)
		os.makedirs(self.temp_path)

		chunk_pos_vel = np.zeros((self.chunk_size, 6, N_x, N_y))
		chunk_forces = np.zeros((self.chunk_size, 3, N_x, N_y))
		self.chunk_counter = 0
		self.current_chunk_size = 0
		for dataset_path in dataset_paths:
			i = 0
			while True:
				try:
					temp_data = np.load(dataset_path + f"/{i}.npy")
					temp_pos = temp_data[:, :3, ...]
					temp_vel = temp_data[:, 3:6, ...]
					temp_forces = temp_data[:, 6:, ...]

					while True:
						if (self.current_chunk_size+temp_pos.shape[0]) <= self.chunk_size:
							start = self.current_chunk_size
							end = self.current_chunk_size+temp_pos.shape[0]
							chunk_pos_vel[start:end, :3, :, :] = temp_pos
							chunk_pos_vel[start:end, 3:, :, :] = temp_vel
							chunk_forces[start:end, :, :, :] = temp_forces

							self.current_chunk_size += temp_pos.shape[0]

							if self.current_chunk_size == self.chunk_size:
								np.save(self.temp_path + f"/pos_vel_{self.chunk_counter}.npy", chunk_pos_vel, allow_pickle=False)
								np.save(self.temp_path + f"/forces_{self.chunk_counter}.npy", chunk_forces, allow_pickle=False)
								self.chunk_counter += 1

								self.current_chunk_size = 0

							break

						else:
							cut = self.chunk_size-self.current_chunk_size
							chunk_pos_vel[-cut:, :3, :, :] = temp_pos[:cut, :]
							chunk_pos_vel[-cut:, 3:, :, :] = temp_vel[:cut, :]
							chunk_forces[-cut:, :, :, :] = temp_forces[:cut, :]
							temp_pos = temp_pos[cut:, :, :, :]
							temp_vel = temp_vel[cut:, :, :, :]
							temp_forces = temp_forces[cut:, :, :, :]

							np.save(self.temp_path + f"/pos_vel_{self.chunk_counter}.npy", chunk_pos_vel, allow_pickle=False)
							np.save(self.temp_path + f"/forces_{self.chunk_counter}.npy", chunk_forces, allow_pickle=False)
							self.chunk_counter += 1

							self.current_chunk_size = 0

					i += 1
					
				except FileNotFoundError:
					break

		if self.current_chunk_size > 0:
			np.save(self.temp_path + f"/pos_vel_{self.chunk_counter}.npy", chunk_pos_vel[:self.current_chunk_size, :, :, :], allow_pickle=False)
			np.save(self.temp_path + f"/forces_{self.chunk_counter}.npy", chunk_forces[:self.current_chunk_size, :, :, :], allow_pickle=False)
			self.chunk_counter += 1