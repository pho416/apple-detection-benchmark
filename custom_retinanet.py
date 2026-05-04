import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetRegressionHead
from torchvision.ops import generalized_box_iou_loss
import timm
from collections import OrderedDict

# Thử import complete_box_iou_loss, nếu torchvision cũ không có thì dùng generalized
try:
    from torchvision.ops import complete_box_iou_loss
except ImportError:
    complete_box_iou_loss = None

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, activation=True):
        super().__init__()
        if padding is None: padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return self.sigmoid(avg + mx) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return attn * x

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()
    def forward(self, x): return self.spatial(self.channel(x))

class IoURetinaNetRegressionHead(RetinaNetRegressionHead):
    def __init__(self, in_channels, num_anchors, loss_type='giou'):
        super().__init__(in_channels, num_anchors)
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.loss_type = loss_type.lower()
    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        pass # Phần loss không cần thiết cho lúc Inference trên web

class CustomRetinaNetHead(RetinaNetHead):
    def __init__(self, in_channels, num_anchors, num_classes, bbox_loss_type='giou'):
        super().__init__(in_channels, num_anchors, num_classes)
        self.regression_head = IoURetinaNetRegressionHead(in_channels, num_anchors, loss_type=bbox_loss_type)

class PANetBackbone(nn.Module):
    def __init__(self, backbone_name='cspdarknet53', out_channels=256, pretrained=False):
        super().__init__()
        self.body = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=(2, 3, 4))
        channels = self.body.feature_info.channels()
        self.lateral3 = ConvBNAct(channels[0], out_channels, 1)
        self.lateral4 = ConvBNAct(channels[1], out_channels, 1)
        self.lateral5 = ConvBNAct(channels[2], out_channels, 1)
        self.smooth3 = ConvBNAct(out_channels, out_channels, 3)
        self.smooth4 = ConvBNAct(out_channels, out_channels, 3)
        self.smooth5 = ConvBNAct(out_channels, out_channels, 3)
        self.down_p3 = ConvBNAct(out_channels, out_channels, 3, stride=2)
        self.down_p4 = ConvBNAct(out_channels, out_channels, 3, stride=2)
        self.pan3 = ConvBNAct(out_channels, out_channels, 3)
        self.pan4 = ConvBNAct(out_channels, out_channels, 3)
        self.pan5 = ConvBNAct(out_channels, out_channels, 3)
        self.cbam3 = CBAM(out_channels)
        self.cbam4 = CBAM(out_channels)
        self.cbam5 = CBAM(out_channels)
        self.out_channels = out_channels
    def forward(self, x):
        c3, c4, c5 = self.body(x)
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        n3 = self.pan3(p3)
        n4 = self.pan4(p4 + self.down_p3(n3))
        n5 = self.pan5(p5 + self.down_p4(n4))
        n3 = self.cbam3(n3)
        n4 = self.cbam4(n4)
        n5 = self.cbam5(n5)
        return OrderedDict([('p3', n3), ('p4', n4), ('p5', n5)])

def build_model(num_classes=3):
    # Định nghĩa cấu trúc chuẩn khớp với file huấn luyện[cite: 1]
    anchor_sizes = ((12, 16, 20), (24, 28, 32), (36, 40, 44))
    anchor_ratios = ((0.83, 1.0, 1.17),) * 3
    backbone = PANetBackbone(backbone_name='cspdarknet53', out_channels=256, pretrained=False)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_ratios)
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    head = CustomRetinaNetHead(in_channels=backbone.out_channels, num_anchors=num_anchors, num_classes=num_classes)
    model = RetinaNet(backbone=backbone, num_classes=num_classes, head=head, anchor_generator=anchor_generator, score_thresh=0.35, nms_thresh=0.50, detections_per_img=300, min_size=640, max_size=640)
    return model