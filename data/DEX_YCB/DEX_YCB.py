import copy
import os

import cv2
import numpy as np

from torch.utils.data import Dataset
from pycocotools.coco import COCO

from config import cfg
from utils.mano import MANO
from utils.preprocessing import (
	augmentation,
	get_bbox,
	load_img,
	process_bbox,
)
from utils.transforms import rigid_align


mano = MANO()

class DEX_YCB(Dataset):
	def __init__(self, transform, data_split):
		self.transform = transform
		self.data_split = data_split if data_split == "train" else "test"
		self.root_dir = os.path.join("..", "data", "DEX_YCB", "data")
		self.annotations_path = os.path.join(self.root_dir, "annotations")
		self.root_joint_idx = 0

		self.data_list = self.load_data()
		if self.data_split != "train":
			self.eval_result = [[], []]  # [mpjpe_list, pa-mpjpe_list]

	def load_data(self):
		db = COCO(os.path.join(self.annotations_path, "DEX_YCB_s0_{}_subset_data.json".format(self.data_split)))
		data_list = []
		for aid in db.anns.keys():
			ann = db.anns[aid]
			image_id = ann["image_id"]
			img = db.loadImgs(image_id)[0]
			img_path = os.path.join(self.root_dir, img["file_name"])
			depth_img_path = img_path.replace("color_", "aligned_depth_to_color_").replace(".jpg", ".png")
			img_shape = (img["height"], img["width"])
			if self.data_split == "train":
				joints_coord_cam = np.array(ann["joints_coord_cam"], dtype=np.float32)  # meter
				cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}
				joints_coord_img = np.array(ann["joints_img"], dtype=np.float32)
				hand_type = ann["hand_type"]

				bbox = get_bbox(
					joints_coord_img[:, :2],
					np.ones_like(joints_coord_img[:, 0]),
					expansion_factor=1.5,
				)
				bbox = process_bbox(bbox, img["width"], img["height"], expansion_factor=1.0)

				if bbox is None:
					continue

				mano_pose = np.array(ann["mano_param"]["pose"], dtype=np.float32)
				mano_shape = np.array(ann["mano_param"]["shape"], dtype=np.float32)

				data = {
					"img_path": img_path,
					"depth_img_path": depth_img_path,
					"img_shape": img_shape,
					"joints_coord_cam": joints_coord_cam,
					"joints_coord_img": joints_coord_img,
					"bbox": bbox,
					"cam_param": cam_param,
					"mano_pose": mano_pose,
					"mano_shape": mano_shape,
					"hand_type": hand_type,
				}
			else:
				joints_coord_cam = np.array(ann["joints_coord_cam"], dtype=np.float32)
				root_joint_cam = copy.deepcopy(joints_coord_cam[0])
				joints_coord_img = np.array(ann["joints_img"], dtype=np.float32)
				hand_type = ann["hand_type"]

				bbox = get_bbox(
					joints_coord_img[:, :2],
					np.ones_like(joints_coord_img[:, 0]),
					expansion_factor=1.5,
				)
				bbox = process_bbox(bbox, img["width"], img["height"], expansion_factor=1.0)
				if bbox is None:
					bbox = np.array([0, 0, img["width"] - 1, img["height"] - 1], dtype=np.float32)

				cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}

				data = {
					"img_path": img_path,
					"depth_img_path": depth_img_path,
					"img_shape": img_shape,
					"joints_coord_cam": joints_coord_cam,
					"root_joint_cam": root_joint_cam,
					"bbox": bbox,
					"cam_param": cam_param,
					"image_id": image_id,
					"hand_type": hand_type,
				}

			data_list.append(data)
		return data_list

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		data = copy.deepcopy(self.data_list[idx])
		img_path, depth_img_path, img_shape, bbox = (
			data["img_path"],
			data["depth_img_path"],
			data["img_shape"],
			data["bbox"],
		)
		hand_type = data["hand_type"]
		do_flip = hand_type == "left"

		# img
		img = load_img(img_path)
		orig_img = copy.deepcopy(img)[:, :, ::-1] # Convert image from BGR to RGB channel
		img, img2bb_trans, bb2img_trans, rot, scale = augmentation(
			img, bbox, self.data_split, do_flip=do_flip
		)
		img = self.transform(img.astype(np.float32)) / 255.0

		# depth image
		depth_img = load_img(depth_img_path)
		# orig_depth_img = copy.deepcopy(depth_img)[:, :, ::-1]
		depth_img, depth_img2bb_trans, bb2depth_trans, rot, scale = augmentation(
			depth_img, bbox, self.data_split, do_flip=do_flip
		)
		depth_img = self.transform(depth_img.astype(np.float32)) / 255.0

		if self.data_split == "train":
			# 2D joint coordinate
			joints_img = data["joints_coord_img"]
			if do_flip:
				joints_img[:, 0] = img_shape[1] - joints_img[:, 0] - 1
			joints_img_xy1 = np.concatenate(
				(joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1
			)
			joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[
				:, :2
			]
			# normalize to [0,1]
			joints_img[:, 0] /= cfg.input_img_shape[1]
			joints_img[:, 1] /= cfg.input_img_shape[0]

			# 3D joint camera coordinate
			joints_coord_cam = data["joints_coord_cam"]
			root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
			# root-relative
			joints_coord_cam -= joints_coord_cam[self.root_joint_idx, None, :]
			if do_flip:
				joints_coord_cam[:, 0] *= -1

			# 3D data rotation augmentation
			rot_aug_mat = np.array(
				[
					[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
					[np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
					[0, 0, 1],
				],
				dtype=np.float32,
			)
			joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(
				1, 0
			)

			# mano parameter
			mano_pose, mano_shape = data["mano_pose"], data["mano_shape"]

			# 3D data rotation augmentation
			mano_pose = mano_pose.reshape(-1, 3)
			if do_flip:
				mano_pose[:, 1:] *= -1
			root_pose = mano_pose[self.root_joint_idx, :]
			root_pose, _ = cv2.Rodrigues(root_pose)
			root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
			mano_pose[self.root_joint_idx] = root_pose.reshape(3)
			mano_pose = mano_pose.reshape(-1)

			inputs = {
				"img": img,
				"depth_img": depth_img
			}
			targets = {
				"joints_img": joints_img,
				"joints_coord_cam": joints_coord_cam,
				"mano_pose": mano_pose,
				"mano_shape": mano_shape,
			}
			meta_info = {"root_joint_cam": root_joint_cam}

		else:
			root_joint_cam = data["root_joint_cam"]
			inputs = {
				"img": img,
				"depth_img": depth_img
			}
			targets = {}
			meta_info = {"root_joint_cam": root_joint_cam}

		return inputs, targets, meta_info

	def evaluate(self, outs, cur_sample_idx):
		annotations = self.data_list
		sample_num = len(outs)
		for n in range(sample_num):
			annotation = annotations[cur_sample_idx + n]
			out = outs[n]
			joints_out = out["joints_coord_cam"]

			# root centered
			joints_out -= joints_out[self.root_joint_idx]

			# flip back to left hand
			if annotation["hand_type"] == "left":
				joints_out[:, 0] *= -1

			# root align
			gt_root_joint_cam = annotation["root_joint_cam"]
			joints_out += gt_root_joint_cam

			# GT and rigid align
			joints_gt = annotation["joints_coord_cam"]
			joints_out_aligned = rigid_align(joints_out, joints_gt)

			# m to mm
			joints_out *= 1000
			joints_out_aligned *= 1000
			joints_gt *= 1000

			self.eval_result[0].append(np.sqrt(np.sum((joints_out - joints_gt) ** 2, 1)).mean())
			self.eval_result[1].append(
				np.sqrt(np.sum((joints_out_aligned - joints_gt) ** 2, 1)).mean()
			)

	def print_eval_result(self, test_epoch):
		print("Epoch %d evaluation result:" % test_epoch)
		print("MPJPE : %.2f mm" % np.mean(self.eval_result[0]))
		print("PA MPJPE : %.2f mm" % np.mean(self.eval_result[1]))
