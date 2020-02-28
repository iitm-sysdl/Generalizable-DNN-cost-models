from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

class BatchNorm2d(nn.Module):
    """
    Fixed version of BatchNorm2d, which has only the scale and bias
    """

    def __init__(self, out):
        super(BatchNorm2d, self).__init__()
        self.register_buffer("scale", torch.ones(out))
        self.register_buffer("bias", torch.zeros(out))

    #@torch.jit.script_method
    def forward(self, x):
        scale = self.scale.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return x * scale + bias


class BiasAdd(nn.Module):
    """
    Fixed version of BatchNorm2d, which has only the scale and bias
    """

    def __init__(self, out):
        super(BiasAdd, self).__init__()
        self.register_buffer("bias", torch.zeros(out))

    #@torch.jit.script_method
    def forward(self, x):
        bias = self.bias.view(1, -1, 1, 1)
        return x + bias


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        #import pdb; pdb.set_trace()
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


def box_area(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def box_iou(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = box_area(overlap_left_top, overlap_right_bottom)
    area0 = box_area(boxes0[..., :2], boxes0[..., 2:])
    area1 = box_area(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def nms(box_scores, iou_threshold):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = box_iou(rest_boxes, current_box.unsqueeze(0))
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


@torch.jit.script
def decode_boxes(rel_codes, boxes, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor

    # perform some unpacking to make it JIT-fusion friendly
    
    #rel_codes=rel_codes[0][None]
    wx = weights[1]
    wy = weights[0]
    ww = weights[3]
    wh = weights[2]

    boxes_x1 = boxes[:, 1].unsqueeze(1).unsqueeze(0)
    boxes_y1 = boxes[:, 0].unsqueeze(1).unsqueeze(0)
    boxes_x2 = boxes[:, 3].unsqueeze(1).unsqueeze(0)
    boxes_y2 = boxes[:, 2].unsqueeze(1).unsqueeze(0)

    dx = rel_codes[:,:, 1].unsqueeze(2)
    dy = rel_codes[:,:, 0].unsqueeze(2)
    dw = rel_codes[:,:, 3].unsqueeze(2)
    dh = rel_codes[:,:, 2].unsqueeze(2)

    # implementation starts here
    widths = boxes_x2 - boxes_x1
    heights = boxes_y2 - boxes_y1
    ctr_x = boxes_x1 + 0.5 * widths
    ctr_y = boxes_y1 + 0.5 * heights

    dx = dx / wx
    dy = dy / wy
    dw = dw / ww
    dh = dh / wh

    pred_ctr_x = dx * widths + ctr_x
    #import pdb; pdb.set_trace()
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.cat(
        [
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ],
        dim=2,
    )
    #import pdb; pdb.set_trace()
    return pred_boxes

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        OrderedDict(
            [
                ("0", Conv2d_tf(inp, oup, 3, stride, padding="SAME", bias=False)),
                ("0/BatchNorm", BiasAdd(oup)),
                ("0/ReLU", nn.ReLU6(inplace=True)),
            ]
        )
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "depthwise",
                    Conv2d_tf(
                        inp, inp, 3, stride, padding="SAME", groups=inp, bias=False
                    ),
                ),
                ("depthwise/BatchNorm", BatchNorm2d(inp)),
                ("depthwise/ReLU", nn.ReLU6(inplace=True)),
                ("pointwise", nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
                ("pointwise/BatchNorm", BiasAdd(oup)),
                ("pointwise/ReLU", nn.ReLU6(inplace=True)),
            ]
        )
    )


class MobileNetV1Base(nn.ModuleList):
    def __init__(self, return_layers=[11, 13]):
        super(MobileNetV1Base, self).__init__(
            [
                conv_bn(3, 32, 2),
                conv_dw(32, 64, 1),
                conv_dw(64, 128, 2),
                conv_dw(128, 128, 1),
                conv_dw(128, 256, 2),
                conv_dw(256, 256, 1),
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1),
            ]
        )
        self.return_layers = return_layers

    def forward(self, x):
        out = []
        for idx, module in enumerate(self):
            x = module(x)
            if idx in self.return_layers:
                out.append(x)
        return out


class PredictionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super(PredictionHead, self).__init__()
        self.classification = nn.Conv2d(
            in_channels, num_classes * num_anchors, kernel_size=1
        )
        self.regression = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        bs = x.shape[0]
        class_logits = self.classification(x)
        box_regression = self.regression(x)

        class_logits = class_logits.permute(0, 2, 3, 1).reshape(
            bs, -1, self.num_classes
        )
        box_regression = box_regression.permute(0, 2, 3, 1).reshape(bs, -1, 4)

        return class_logits, box_regression


class Block(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Block, self).__init__(
            nn.Conv2d(in_channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU6(),
            Conv2d_tf(
                mid_channels, out_channels, kernel_size=3, stride=2, padding="SAME"
            ),
            nn.ReLU6(),
        )


class SSD(nn.Module):
    def __init__(self, backbone, extras):
        super(SSD, self).__init__()

        self.backbone = backbone
        self.extras = extras
        #self.predictors = predictors

        # preprocess
        self.image_size = 300
        self.image_mean = 127.5
        self.image_std = 127.5

        self.coder_weights = torch.tensor((10, 10, 5, 5), dtype=torch.float32)
        self._feature_map_shapes = None

        # postprocess
        self.nms_threshold = 0.6

        # set it to 0.01 for better results but slower runtime
        self.score_threshold = 0.3

    def ssd_model(self, x):
        feature_maps = self.backbone(x)

        out = feature_maps[-1]
        for module in self.extras:
            out = module(out)
            feature_maps.append(out)

        # results = []
        # for feature, module in zip(feature_maps, self.predictors):
        #     results.append(module(feature))

        # class_logits, box_regression = list(zip(*results))
        # class_logits = torch.cat(class_logits, 1)
        # box_regression = torch.cat(box_regression, 1)

        # scores = torch.sigmoid(class_logits)
        # box_regression = box_regression.squeeze(0)

        # shapes = [o.shape[-2:] for o in feature_maps]
        # if shapes != self._feature_map_shapes:
        #     # generate anchors for the sizes of the feature map
        #     priors = create_ssd_anchors()._generate(shapes)
        #     priors = torch.cat(priors, dim=0)
        #     self.priors = priors.to(scores)
        #     self._feature_map_shapes = shapes

        # self.coder_weights = self.coder_weights.to(scores)
        # if box_regression.dim()==2:
        #     box_regression = box_regression[None]
        # boxes = decode_boxes(box_regression, self.priors, self.coder_weights)
        # add a batch dimension
        return feature_maps[0]


    def forward(self, images):
        """
        Arguments:
            images (torch.Tensor[N,C,H,W]):
        """

        # scores, boxes = 
        # list_boxes=[]; list_labels=[]; list_scores=[]
        # for b in range(len(scores)):
        #     bboxes, blabels, bscores = self.filter_results(scores[b], boxes[b])
        #     list_boxes.append(bboxes)
        #     list_labels.append(blabels.long())
        #     list_scores.append(bscores)
        # #boxes = self.rescale_boxes(boxes, height, width)
        return self.ssd_model(images)


def create_mobilenetv1_ssd(num_classes):
    backbone = MobileNetV1Base()

    extras = nn.ModuleList(
        [
            Block(1024, 256, 512),
            Block(512, 128, 256),
            Block(256, 128, 256),
            Block(256, 64, 128),
        ]
    )

    # predictors = nn.ModuleList(
    #     [
    #         PredictionHead(in_channels, num_classes, num_anchors)
    #         for in_channels, num_anchors in zip(
    #             (512, 1024, 512, 256, 256, 128), (3, 6, 6, 6, 6, 6)
    #         )
    #     ]
    # )

    return SSD(backbone, extras)


def get_tf_pretrained_mobilenet_ssd(weights_file):
    from models.convert_tf_weights import get_state_dict, read_tf_weights

    model = create_mobilenetv1_ssd(91)
    weights = read_tf_weights(weights_file)
    state_dict = get_state_dict(weights)
    model.load_state_dict(state_dict)
    return model
