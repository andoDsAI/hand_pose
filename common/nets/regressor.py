import torch.nn as nn
from nets.hand_head import HandEncoder, HandRegHead
from nets.mano_head import ManoRegHead


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.hand_regHead = HandRegHead()
        self.hand_encoder = HandEncoder()
        self.mano_regHead = ManoRegHead()

    def forward(self, feats, gt_mano_params=None):
        out_hm, encoding, preds_joints_img = self.hand_regHead(feats)
        mano_encoding = self.hand_encoder(out_hm, encoding)
        pred_mano_results, gt_mano_results = self.mano_regHead(mano_encoding, gt_mano_params)
        return pred_mano_results, gt_mano_results, preds_joints_img
