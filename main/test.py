import argparse

import torch
import torch.backends.cudnn as cudnn
from accelerate import Accelerator, set_seed

from config import cfg
from base import Tester
from tqdm import tqdm


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu", type=str, dest="gpu_ids")
	parser.add_argument("--seed", default=42, type=int, help="Seed.")
	parser.add_argument("--test_epoch", type=str, dest="test_epoch")
	args = parser.parse_args()

	if not args.gpu_ids:
		assert 0, "Please set proper gpu ids"

	if "-" in args.gpu_ids:
		gpus = args.gpu_ids.split("-")
		gpus[0] = int(gpus[0])
		gpus[1] = int(gpus[1]) + 1
		args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

	assert args.test_epoch, "Test epoch is required."
	return args


def main():
	# argument parse and create log
	args = parse_args()
	set_seed(args.seed)
	cfg.set_args(args.gpu_ids)
	cudnn.benchmark = True
	
	# hugging-face Accelerator
	accelerator = Accelerator()

	tester = Tester(args.test_epoch)
	tester.initialize()
 
	model, dataloader = accelerator.prepare(tester.model, tester.dataloader)

	cur_sample_idx = 0
	model.eval()
	for inputs, targets, meta_info in tqdm(dataloader):
		# forward
		with torch.no_grad():
			out = model(inputs, targets, meta_info, "test")

		# save output
		out = {k: v.cpu().numpy() for k, v in out.items()}
		for k, _ in out.items():
			batch_size = out[k].shape[0]
		out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

		# evaluate
		tester._evaluate(out, cur_sample_idx)
		cur_sample_idx += len(out)

	tester._print_eval_result(args.test_epoch)


if __name__ == "__main__":
	main()
