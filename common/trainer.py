import glob
import math
import os

import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader

from config import cfg
from base import Base
from nets.network import get_model


# import dataset class
exec("from " + cfg.train_set + " import " + cfg.train_set)


class Trainer(Base):
	def __init__(self, log_name="train_logs.txt"):
		super(Trainer, self).__init__(log_name=log_name)

	def get_optimizer(self, model):
		model_params = filter(lambda p: p.requires_grad, model.parameters())
		optimizer = torch.optim.Adam(model_params, lr=cfg.lr)
		return optimizer

	def save_model(self, state, epoch):
		file_path = os.path.join(cfg.model_dir, "snapshot_{}.pth.tar".format(str(epoch)))
		torch.save(state, file_path)
		self.logger.info("Write snapshot into {}".format(file_path))

	def load_model(self, model, optimizer):
		model_file_list = glob.glob(os.path.join(cfg.model_dir, "*.pth.tar"))
		cur_epoch = max(
			[
				int(file_name[file_name.find("snapshot_") + 9: file_name.find(".pth.tar")])
				for file_name in model_file_list
			]
		)
		checkpoint_path = os.path.join(cfg.model_dir, "snapshot_" + str(cur_epoch) + ".pth.tar")
		checkpoint = torch.load(checkpoint_path)
		start_epoch = checkpoint["epoch"] + 1
		model.load_state_dict(checkpoint["network"], strict=False)
		optimizer.load_state_dict(checkpoint['optimizer'])

		self.logger.info("Load checkpoint from {}".format(checkpoint_path))
		return start_epoch, model, optimizer

	def set_lr(self, epoch):
		for e in cfg.lr_dec_epoch:
			if epoch < e:
				break
		if epoch < cfg.lr_dec_epoch[-1]:
			idx = cfg.lr_dec_epoch.index(e)
			for g in self.optimizer.param_groups:
				g["lr"] = cfg.lr * (cfg.lr_dec_factor**idx)
		else:
			for g in self.optimizer.param_groups:
				g["lr"] = cfg.lr * (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

	def get_lr(self):
		for g in self.optimizer.param_groups:
			cur_lr = g["lr"]
		return cur_lr

	def _make_batch_generator(self):
		# data load and construct batch generator
		self.logger.info("Creating dataset...")
		dataset = eval(cfg.train_set)(transforms.ToTensor(), "train")
		self.dataset = dataset
		self.itr_per_epoch = math.ceil(len(dataset) / cfg.num_gpus / cfg.train_batch_size)
		self.dataloader = DataLoader(
			dataset=dataset,
			batch_size=cfg.num_gpus * cfg.train_batch_size,
			shuffle=True,
			num_workers=cfg.num_thread,
			pin_memory=True,
		)

	def _make_model(self):
		# prepare network
		self.logger.info("Creating graph and optimizer...")
		model = get_model("train")

		model = DataParallel(model).cuda()
		optimizer = self.get_optimizer(model)
		if cfg.continue_train:
			start_epoch, model, optimizer = self.load_model(model, optimizer)
		else:
			start_epoch = 0

		self.start_epoch = start_epoch
		self.model = model
		self.optimizer = optimizer

	def _get_evaluate_result(self):
		return self.dataset.get_eval_result()
