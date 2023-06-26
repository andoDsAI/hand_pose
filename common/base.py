import abc

from config import cfg
from logger import ColorLogger
from timer import Timer


# import dataset class
exec("from " + cfg.train_set + " import " + cfg.train_set)
exec("from " + cfg.test_set + " import " + cfg.test_set)


class Base(metaclass=abc.ABCMeta):
	def __init__(self, log_name="logs.txt"):
		self.cur_epoch = 0
		self.dataset = None
		self.data_loader = None

		# timer
		self.tot_timer = Timer()
		self.gpu_timer = Timer()
		self.read_timer = Timer()

		# logger
		self.log_name = log_name
		self.logger = ColorLogger(cfg.log_dir, log_name=log_name)

	@abc.abstractmethod
	def _make_batch_generator(self):
		raise NotImplementedError

	@abc.abstractmethod
	def _make_model(self):
		raise NotImplementedError

	def _evaluate(self, outs, cur_sample_idx):
		eval_result = self.dataset.evaluate(outs, cur_sample_idx)
		return eval_result

	def compile(self):
		self._make_batch_generator()
		self._make_model()
