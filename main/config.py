import os
import sys


class Config(object):
    # dataset
    train_set = "DEX_YCB"
    test_set = "DEX_YCB"
    use_full_data = True
    use_depth_img = False
 
    # network
    architecture = "default"
    backend = "resnet18"

    ## input, output shape
    input_img_shape = (256, 256)

    # training config
    end_epoch = 40
    lr_dec_epoch = [i for i in range(1, end_epoch)]
    lr = 1e-4
    lr_dec_factor = 0.9
    train_batch_size = 32  # per GPU
    gradient_accumulation_steps = 2
    
    # mano config
    lambda_mano_verts = 1e4
    lambda_mano_joints = 1e4
    lambda_mano_pose = 10
    lambda_mano_shape = 0.1
    lambda_joints_img = 100
 
    # save checkpoint config
    checkpoint_freq = 10

    # testing config
    test_batch_size = 32

    # others
    num_thread = 12
    gpu_ids = "0"
    num_gpus = 1
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

    def set_args(self, gpu_ids, continue_train=False, gradient_accumulation_steps=32):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(","))
        self.continue_train = continue_train
        self.gradient_accumulation_steps = gradient_accumulation_steps
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print(">>> Using GPU: {}".format(self.gpu_ids))


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
