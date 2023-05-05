import os
import sys


class Config:
	# dataset
	train_set = "DEX_YCB"
	test_set = "DEX_YCB"

	## input, output shape
	input_img_shape = (256, 256)

	# training config
	end_epoch = 40
	epochs = [i for i in range(1, end_epoch)]
	lr = 1e-4
	lr_dec_factor = 0.9
	train_batch_size = 2  # per GPU
	gradient_accumulation_steps = 32
	
	# mano config
	lambda_mano_verts = 1e4
	lambda_mano_joints = 1e4
	lambda_mano_pose = 10
	lambda_mano_shape = 0.1
	lambda_joints_img = 100
 
	# save checkpoint config
	checkpoint_freq = 10

	# testing config
	test_batch_size = 8

	# others
	num_thread=12
	continue_train = False

	# directory
	cur_dir = os.path.dirname(os.path.abspath(__file__))
	root_dir = os.path.join(cur_dir, "..")
	data_dir = os.path.join(root_dir, "data")
	output_dir = os.path.join(root_dir, "output")
	model_dir = os.path.join(output_dir, "model_dump")
	vis_dir = os.path.join(output_dir, "vis")
	log_dir = os.path.join(output_dir, "log")
	result_dir = os.path.join(output_dir, "result")
	mano_path = os.path.join(root_dir, "common", "utils", "manopth")

	def set_args(self, continue_train=False, gradient_accumulation_steps=32):
		self.continue_train = continue_train
		self.gradient_accumulation_steps = gradient_accumulation_steps


cfg = Config()

sys.path.insert(0, os.path.join(cfg.root_dir, "common"))
from utils.dir import add_path, make_folder

add_path(os.path.join(cfg.data_dir))
add_path(os.path.join(cfg.data_dir, cfg.train_set))
add_path(os.path.join(cfg.data_dir, cfg.test_set))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
