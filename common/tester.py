import os

import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader

from config import cfg
from base import Base
from nets.network import get_model


# import dataset class
exec("from " + cfg.test_set + " import " + cfg.test_set)


class Tester(Base):
	def __init__(self, test_epoch):
		self.test_epoch = int(test_epoch)
		super(Tester, self).__init__(log_name="test_logs.txt")

	def _make_batch_generator(self):
		# data load and construct batch generator
		self.logger.info("Creating dataset...")
		self.dataset = eval(cfg.test_set)(transforms.ToTensor(), "test")
		self.dataloader = DataLoader(
			dataset=self.dataset,
			batch_size=cfg.num_gpus * cfg.test_batch_size,
			shuffle=False,
			num_workers=cfg.num_thread,
			pin_memory=True,
		)

	def _make_model(self):
		model_path = os.path.join(cfg.model_dir, "snapshot_%d.pth.tar" % self.test_epoch)
		assert os.path.exists(model_path), "Cannot find model at " + model_path
		self.logger.info("Load checkpoint from {}".format(model_path))

		# prepare network
		self.logger.info("Creating graph...")
		model = get_model("test")
		model = DataParallel(model).cuda()
		checkpoint = torch.load(model_path)
		model.load_state_dict(checkpoint["network"], strict=False)

		self.model = model

	def _print_eval_result(self, test_epoch):
		self.dataset.print_eval_result(test_epoch)