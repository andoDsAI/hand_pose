import argparse

import torch
from accelerate import Accelerator

from config import cfg
from base import Tester
from tqdm import tqdm


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_epoch", type=str, dest="test_epoch")

	args = parser.parse_args()
	assert args.test_epoch, "Test epoch is required."
	return args


def main():
	# argument parse and create log
	args = parse_args()

	# hugging-face Accelerator
	accelerator = Accelerator()

	tester = Tester(args.test_epoch)
	tester.initialize()
	model, dataloader = accelerator.prepare(tester.model, tester.dataloader)

	model.eval()
	cur_sample_idx = 0
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
