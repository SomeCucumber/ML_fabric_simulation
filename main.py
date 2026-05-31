# Math:
import numpy as np

# OpenGL:
from opengl.render import load_shader
from opengl.render.meshes import FabricMesh
from opengl.window.window import main

# Simulation:
from simulation import Simulation

# Misc:
import json

# ===== PARAMETERS =====
with open("config/settings.json") as f:
	settings = json.load(f)

resolution = settings["resolution"]
simulation_speed = settings["simulation_speed"]
benchmark_start_stop = settings["benchmark_start_stop"]
headless = settings["headless"]
# ======================

# Initialize simulation:
sim = Simulation()
x_scale, y_scale, x_offset, y_offset = sim.get_transform()

if headless:
	while True:
		sim.step()

else:
	main(
		sim,
		shader_loader=load_shader,
		shader_name="fabric",
		mesh=FabricMesh,
		window_res=resolution,
		simulation_speed=simulation_speed,
		benchmark_start_stop=benchmark_start_stop,
		transform=(x_scale, y_scale, x_offset, y_offset),
		window_name="Fabric Simulation"
		)