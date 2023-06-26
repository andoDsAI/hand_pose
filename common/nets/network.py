import torch
import torch.nn as nn
from torch.nn import functional as F


from config import cfg
from nets.fpn import FPN
from nets.regressor import Regressor
from nets.transformer import Transformer
from nets.pointnet import PointNetFeat


fpn_models = {
	"resnet18": FPN(fpn_size=512, lateral_sizes=[256, 128, 64], deep_feature_size=256, backend="resnet18", pretrained=True),
	"resnet34": FPN(fpn_size=512, lateral_sizes=[256, 128, 64], deep_feature_size=256, backend="resnet34", pretrained=True),
	"resnet50": FPN(fpn_size=2048, lateral_sizes=[1024, 512, 256], deep_feature_size=256, backend="resnet50", pretrained=True),
	"resnet101": FPN(fpn_size=2048, lateral_sizes=[1024, 512, 256], deep_feature_size=256, backend="resnet101", pretrained=True),
	"resnet152": FPN(fpn_size=2048, lateral_sizes=[1024, 512, 256], deep_feature_size=256, backend="resnet152", pretrained=True),
}


class HandPoseNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module = FPN(pretrained=True),
        point_net: nn.Module = PointNetFeat(global_feat=False),
        FIT: nn.Module = Transformer(injection=True),
        SET: nn.Module = Transformer(injection=False),
        regressor: nn.Module = Regressor()
    ):
        super(HandPoseNet, self).__init__()
        self.backbone = backbone
        self.pointnet = point_net
        self.FIT = FIT
        self.SET = SET
        self.regressor = regressor
        self.conv1 = nn.Conv2d(1280, 256, 1)
    
    def forward(self, inputs, targets, meta_info, mode):
        # get primary, secondary features
        p_feats, s_feats = self.backbone(inputs["img"])

        # # get depth features
        # depth_feats, _, _ = self.pointnet(inputs["depth_img"])

        # # combine depth features to other features
        # p_feats = torch.cat([p_feats, depth_feats], dim=1)  # concat depth features to primary features
        # p_feats = F.relu(self.conv1(p_feats))

        # s_feats = torch.cat([s_feats, depth_feats], dim=1)  # concat depth features to secondary features
        # s_feats = F.relu(self.conv1(s_feats))
        
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
            # predict output
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


def get_model(mode: str = "train") -> nn.Module:
    backbone = fpn_models[cfg.backend]
    point_net = PointNetFeat(global_feat=False, feature_transform=False)
    FIT = Transformer(injection=True)   # feature injecting transformer
    SET = Transformer(injection=False)  # self enhancing transformer
    regressor = Regressor()
    
    if mode == "train":
        point_net.apply(init_weights)
        FIT.apply(init_weights)
        SET.apply(init_weights)
        regressor.apply(init_weights)
    
    model = HandPoseNet(backbone, point_net, FIT, SET, regressor)
    return model
