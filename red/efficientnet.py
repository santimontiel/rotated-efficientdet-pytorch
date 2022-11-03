import torch.nn as nn
import math
import yaml
from yaml import SafeLoader
from red.blocks import ConvBlock, MBConvN


class EfficientNet(nn.Module):

    def __init__(
        self, model: str, in_channels: int = 3, n_classes: int = 0
    ) -> None:
        super().__init__()
        available_models = ["b0"]
        if model not in available_models:
            raise NotImplementedError(f"{model} not available. Try with one of {available_models}.")
        if not isinstance(in_channels, int) or not in_channels > 0:
            raise ValueError(f"Input channels are {in_channels}. It must be a positive integer.")
        if not isinstance(n_classes, int) or not n_classes >= 0:
            raise ValueError(f"Number of classes is {n_classes}. It must be 0 for only backbone or \
                positive for classification tasks.")

        self.backbone_cfg = self._load_config(model)
        self.features = self._build_model(in_channels, self.backbone_cfg)
        self.head = self._build_head(n_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.features(x)
        x = self.head(x) 
        return x


    def _build_model(self, channels, config):
        layers = nn.Sequential()
        for i, stage in enumerate(config):
            name, se, t, k, s, chn, l = stage
            if name == "ConvBlock":
                in_chn = channels if i == 0 else config[i-1][5]
                layers.append(ConvBlock(in_chn, chn, k, s))
            elif name == "MBConv":
                this_stage = nn.Sequential()
                for j in range(l):
                    in_chn = config[i-1][5] if j == 0 else chn            
                    this_stage.append(MBConvN(in_chn, chn, k, s, t))
                layers.append(this_stage)
        return layers


    def _build_head(self, n_classes):
        if n_classes == 0:
            return nn.Identity()
        else:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.backbone_cfg[-1][5], n_classes)
            )


    def _load_config(self, model):
        file_name = "effnet_" + model + ".yaml"
        with open("/home/robesafe/workspace_2023/projects/rotated-efficientdet-pytorch/red/models/" + file_name, "r") as stream:
            config = yaml.load(stream, Loader=SafeLoader)
        return config["backbone"]