import torch
import torch.nn as nn
from torch.nn import functional as F

from config import cfg
from nets.backbone import FPN
from nets.regressor import Regressor
from nets.transformer import Transformer
from nets.pointnet import PointNetFeat
from utils.mano import MANO


class Model(nn.Module):
    def __init__(self, backbone, FIT, SET, regressor):
        super(Model, self).__init__()
        self.backbone = backbone
        self.pointnet = PointNetFeat(global_feat=False)
        self.FIT = FIT
        self.SET = SET
        self.conv1 = nn.Conv2d(1280, 256, 1)
        self.regressor = regressor

    def forward(self, inputs, targets, meta_info, mode):
        # primary, secondary feats
        p_feats, s_feats = self.backbone(inputs["img"])
        
        # depth features
        pointnet_feats, _, _ = self.pointnet(inputs["depth_img"])
        # concat depth features to secondary features
        s_feats = torch.cat([s_feats, pointnet_feats], dim=1)
        s_feats = F.relu(self.conv1(s_feats))
		# concat depth features to primary features
        p_feats = torch.cat([p_feats, pointnet_feats], dim=1)
        p_feats = F.relu(self.conv1(p_feats))

        feats = self.FIT(s_feats, p_feats)
        feats = self.SET(feats, feats)

        if mode == "train":
            gt_mano_params = torch.cat([targets["mano_pose"], targets["mano_shape"]], dim=1)
        else:
            gt_mano_params = None
        pred_mano_results, gt_mano_results, preds_joints_img = self.regressor(
            feats, gt_mano_params
        )

        if mode == "train":
            # loss functions
            loss = {}
            loss["mano_verts"] = cfg.lambda_mano_verts * F.mse_loss(
                pred_mano_results["verts3d"], gt_mano_results["verts3d"]
            )
            loss["mano_joints"] = cfg.lambda_mano_joints * F.mse_loss(
                pred_mano_results["joints3d"], gt_mano_results["joints3d"]
            )
            loss["mano_pose"] = cfg.lambda_mano_pose * F.mse_loss(
                pred_mano_results["mano_pose"], gt_mano_results["mano_pose"]
            )
            loss["mano_shape"] = cfg.lambda_mano_shape * F.mse_loss(
                pred_mano_results["mano_shape"], gt_mano_results["mano_shape"]
            )
            loss["joints_img"] = cfg.lambda_joints_img * F.mse_loss(
                preds_joints_img[0], targets["joints_img"]
            )
            return loss

        else:
            # test output
            out = {}
            out["joints_coord_cam"] = pred_mano_results["joints3d"]
            out["mesh_coord_cam"] = pred_mano_results["verts3d"]
            return out


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def get_model(mode):
    backbone = FPN(pretrained=True)
    FIT = Transformer(injection=True)	# feature injecting transformer
    SET = Transformer(injection=False)	# self enhancing transformer
    regressor = Regressor()

    if mode == "train":
        FIT.apply(init_weights)
        SET.apply(init_weights)
        regressor.apply(init_weights)

    model = Model(backbone, FIT, SET, regressor)
    return model
