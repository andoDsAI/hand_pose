import argparse

from accelerate import Accelerator, set_seed

from config import cfg
from base import Trainer
from tqdm import tqdm


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--continue", dest="continue_train", action="store_true", help="Continue training: load the latest model.")
	parser.add_argument("--gradient_accumulation_steps", default=32, type=int, help="Gradient accumulation steps")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument("--seed", default=42, type=int, help="Seed.")
	parser.add_argument("--log_steps", default=100, type=int, help="Logging steps")
	args = parser.parse_args()

	return args


def main():
	# argument parse and create log
	args = parse_args()
	set_seed(args.seed
	cfg.set_args(args.continue_train, args.gradient_accumulation_steps)
	
	# hugging-face Accelerator
	accelerator = Accelerator(
		device_placement=True,
		gradient_accumulation_steps=args.gradient_accumulation_steps
	)

	trainer = Trainer()
	trainer.initialize()
	model, optimizer, dataloader, lr_scheduler = accelerator.prepare(trainer.model, trainer.optimizer, trainer.dataloader, trainer.lr_scheduler)
 
	# Register the LR scheduler
	accelerator.register_for_checkpointing(lr_scheduler)

	# train
	for epoch in range(trainer.start_epoch, cfg.end_epoch):
		model.train()
		trainer.tot_timer.tic()
		trainer.read_timer.tic()
		for itr, batch in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}/{cfg.end_epoch}: "):
			with accelerator.accumulate(model):
				trainer.read_timer.toc()
				trainer.gpu_timer.tic()

				# input data
				inputs, targets, meta_info = batch

				# forward
				loss = model(inputs, targets, meta_info, "train")
				loss = {k: loss[k].mean() for k in loss}

				# backward
				total_loss = sum(loss[k] for k in loss)
				accelerator.backward(total_loss)
				if accelerator.sync_gradients:
					accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				optimizer.zero_grad()
				optimizer.step()
				
				# logging
				trainer.gpu_timer.toc()
				screen = [
					"Epoch %d/%d iter: %d/%d" % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
					"lr: %g" % (optimizer.param_groups[0]['lr']),
					"speed: %.2f(%.2fs r%.2f)s/itr"
					% (
						trainer.tot_timer.average_time,
						trainer.gpu_timer.average_time,
						trainer.read_timer.average_time,
					),
					"%.2fh/epoch" % (trainer.tot_timer.average_time / 3600.0 * trainer.itr_per_epoch),
				]
				screen += ["%s: %.4f" % ("loss_" + k, v.detach()) for k, v in loss.items()]

				if itr % args.log_steps == 0 or itr == len(dataloader) - 1:
					trainer.logger.info(" ".join(screen))

				trainer.tot_timer.toc()
				trainer.tot_timer.tic()
				trainer.read_timer.tic()

		lr_scheduler.step()
		if (epoch + 1) % cfg.checkpoint_freq == 0 or epoch + 1 == cfg.end_epoch:
			trainer.save_model(
				state={
					"epoch": epoch,
					"network": accelerator.get_state_dict(model),
					"optimizer": accelerator.get_state_dict(optimizer),
					"lr_scheduler": accelerator.get_state_dict(lr_scheduler)
				},
				epoch=epoch + 1,
			)

	accelerator.end_training()

if __name__ == "__main__":
	main()
