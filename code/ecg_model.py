import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW
import pdb
#from ecg_resnet import ECGResNet
from sklearn.metrics import roc_auc_score
from typing import Type, Any, Callable, Union, List, Optional, Iterable
from torch import Tensor
import math
from enum import Enum

device = "cuda" if torch.cuda.is_available() else "cpu"


class ECGLightningModel(pl.LightningModule):
    def __init__(
        self,
        model_fun,
        # batch_size,
        # num_samples,
        lr=0.001,
        wd=0.001,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        finetuning=False,
        **kwargs
    ):
        """
        Args:
            create_model: function that returns torch model that is used for training
            batch_size: batch size
            lr: learning rate
            loss_fn : loss function
            opt: optimizer

        """

        super(ECGLightningModel, self).__init__()
        self.save_hyperparameters(logger=True)
        self.model = model_fun()
        self.val_macro = 0.0
        self.test_macro = 0.0
        self.epoch = 0
        self.finetuning = finetuning
        self.regression = (loss_fn == F.mse_loss)

    def configure_optimizers(self):
        optimizer = self.hparams.opt(self.model.parameters(
        ), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return optimizer

    def forward(self, x, eval=False):
        return self.model(x.float())

    def training_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        loss = self.hparams.loss_fn(preds, targets.float())
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        epoch_loss = torch.tensor(
            [get_loss_from_output(output, key="loss") for output in outputs]
        ).mean()
        self.log("train/total_loss", epoch_loss, on_step=False, on_epoch=True)
        self.log("lr", self.hparams.lr, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        loss = self.hparams.loss_fn(preds, targets.float())
        results = {
            "val_loss": loss,
            "preds": preds.cpu(),
            "targets": targets.cpu(),
        }
        return results

    def test_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        loss = self.hparams.loss_fn(preds, targets.float())
        results = {
            "test_loss": loss,
            "preds": preds.cpu(),
            "targets": targets.cpu(),
        }
        return results

    def validation_epoch_end(self, outputs):
        # outputs[0] when using multiple datasets
        preds = torch.sigmoid(cat(outputs, "preds")).numpy()
        targets = cat(outputs, "targets").numpy()
        val_loss = mean(outputs, "val_loss")
        log = {
            "val/total_loss": val_loss,
        }
        if not self.regression:
            if (targets.sum(axis=0) > 0).all():
                # roc_auc_score crashes in case that not every class is present in at least one sample
                # thats why we skip auc calculation in the sanity check
                macro = roc_auc_score(targets, preds)
            else:
                macro = 0.0
            self.val_macro = macro
            log["val/val_macro"] = macro

        self.log_dict(log)
        return {"val_loss": val_loss, "log": log, "progress_bar": log}

    def test_epoch_end(self, outputs):
        # outputs[0] when using multiple datasets
        preds = torch.sigmoid(cat(outputs, "preds")).numpy()
        targets = cat(outputs, "targets").numpy()
        test_loss = mean(outputs, "test_loss")

        log = {
            "test/total_loss": test_loss,
        }

        if not self.regression:
            macro = roc_auc_score(targets, preds)
            self.test_macro = macro
            log["test/test_macro"] = macro
        
        self.log_dict(log)

        return {"test_loss": test_loss, "log": log, "progress_bar": log}

    def on_train_start(self):
        self.epoch = 0

    def on_train_epoch_end(self):
        self.epoch += 1

    def on_train_epoch_start(self):
        if self.finetuning:
            self.model.eval()
            self.model.freeze_backbone()

    def load_from_checkpoint(self, checkpoint_path, modelname='lenet', lead_order=None):
        lightning_state_dict = torch.load(checkpoint_path)
        state_dict = lightning_state_dict["state_dict"]
        
        print("load model..")
        fails = []
        for name, param in self.named_parameters():
            if param.data.shape == state_dict[name].data.shape:
                param.data = state_dict[name].data
            else:
                fails.append(name)
        for name, param in self.named_buffers():
            if param.data.shape == state_dict[name].data.shape:
                param.data = state_dict[name].data
            else:
                fails.append(name)
        if len(fails) == 0:
            print("all parameters have same shape")
        else:
            print("there has been {} mismatched parameters:".format(len(fails)), fails)
        
def get_loss_from_output(out, key="minimize"):
    return out[key] if isinstance(out, dict) else get_loss_from_output(out[0], key)


def mean(res, key1, key2=None):
    if key2 is not None:
        return torch.stack([x[key1][key2] for x in res]).mean()
    return torch.stack(
        [x[key1] for x in res if type(x) == dict and key1 in x.keys()]
    ).mean()


def cat(res, key):
    return torch.cat(
        [x[key] for x in res if type(x) == dict and key in x.keys()]
    )


def get_model(modelname, num_classes, loss_fn=F.binary_cross_entropy_with_logits):
    def create_lenet_function(num_classes):
        def lenet():
            class LeNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    bn = True
                    ps = False
                    hidden = False
                    c1 = 12  # b/c single time-series
                    c2 = 32  # 4
                    c3 = 64  # 16
                    c4 = 128  # 32
                    k = 5  # kernel size #7
                    s = 2  # stride #3
                    E = 128  # 128
                    modules = []
                    modules.append(nn.Conv1d(c1, c2, k, s))
                    if bn:
                        modules.append(nn.BatchNorm1d(c2))
                    modules.append(nn.ReLU())
                    modules.append(nn.MaxPool1d(2))
                    if ps:
                        modules.append(nn.Dropout(0.1))
                    modules.append(nn.Conv1d(c2, c3, k, s))
                    if bn:
                        modules.append(nn.BatchNorm1d(c3))
                    modules.append(nn.ReLU())
                    modules.append(nn.MaxPool1d(2))
                    if ps:
                        modules.append(nn.Dropout(0.1))
                    modules.append(nn.Conv1d(c3, c4, k, s))
                    if bn:
                        modules.append(nn.BatchNorm1d(c4))
                    modules.append(nn.ReLU())
                    # modules.append(nn.MaxPool1d(2))
                    modules.append(nn.AdaptiveAvgPool1d(1))
                    if ps:
                        modules.append(nn.Dropout(0.1))
                    modules.append(nn.Flatten())
                    modules.append(nn.Linear(c4, E))  # 96
                    modules.append(nn.ReLU())
                    if hidden:
                        modules.append(nn.Dropout(0.5))
                        modules.append(nn.Linear(E, E))
                        modules.append(nn.ReLU())
                        modules.append(nn.Dropout(0.5))
                    modules.append(nn.Linear(E, num_classes))
                    #modules.append(nn.Sigmoid())
                    self.sequential = nn.Sequential(*modules)

                def forward(self, x):
                    return self.sequential(x)

                def freeze_backbone(self):
                    for param in self.sequential.parameters():
                        param.requires_grad = False
                    for param in self.sequential[-1].parameters():
                        param.requires_grad = True

            return LeNet()
        return lenet

    def create_resnet():
        return resnet50(num_classes=num_classes)
        

    def create_xresnet():
        return ECGResNet('xresnet1d50', num_classes)
    
    if modelname == 'lenet':
        model_fun = create_lenet_function(num_classes)
    elif modelname == 'resnet':
        model_fun = create_resnet
    elif modelname == 'xresnet':
        model_fun = create_xresnet
    else:
        raise Exception('model not defined')

    pl_model = ECGLightningModel(model_fun, loss_fn=loss_fn)

    return pl_model


def load_model(modeltype, task, num_classes, path='../output/'):

    modelname = path+modeltype+'_'+task+'/checkpoints/best_model.ckpt'
    model = get_model(modeltype,num_classes)
    model.load_from_checkpoint(modelname)
    if modeltype == 'lenet':
        model.eval()
        model = model.model
        model.to('cuda')
        
    elif modeltype in ['resnet', 'xresnet']:
        model.to('cuda')
        model.eval()

    return model




__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4 # was 4 before because layer1 dim was 64 and layer4 256

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(
            12, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 128, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)




NormType = Enum('NormType', 'Batch BatchZero')


def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None, layer_norm=False):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in) if layer_norm is False else nn.LayerNorm(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

# Cell

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

def _conv1d(in_planes,out_planes,kernel_size=3, stride=1, dilation=1, act="relu", bn=True, drop_p=0, layer_norm=False):
    lst=[]
    if(drop_p>0):
        lst.append(nn.Dropout(drop_p))
    lst.append(nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, dilation=dilation, bias=not(bn)))
    if(bn):
        if(layer_norm):
            lst.append(LambdaLayer(lambda x: x.transpose(1,2)))
            lst.append(nn.LayerNorm(out_planes))
            lst.append(LambdaLayer(lambda x: x.transpose(1,2)))
        else:
            lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    if(act=="gelu"):
        lst.append(nn.GELU())
    return nn.Sequential(*lst)

def _fc(in_planes,out_planes, act="relu", bn=True):
    lst = [nn.Linear(in_planes, out_planes, bias=not(bn))]
    if(bn):
        lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    return nn.Sequential(*lst)

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

# Cell
class SqueezeExcite1d(nn.Module):
    '''squeeze excite block as used for example in LSTM FCN'''
    def __init__(self,channels,reduction=16):
        super().__init__()
        channels_reduced = channels//reduction
        self.w1 = torch.nn.Parameter(torch.randn(channels_reduced,channels).unsqueeze(0))
        self.w2 = torch.nn.Parameter(torch.randn(channels, channels_reduced).unsqueeze(0))

    def forward(self, x):
        #input is bs,ch,seq
        z=torch.mean(x,dim=2,keepdim=True)#bs,ch
        # intermed = F.relu(torch.matmul(self.w1,z))#(1,ch_red,ch * bs,ch,1) = (bs, ch_red, 1)
        intermed = nn.ReLU()(torch.matmul(self.w1,z))#(1,ch_red,ch * bs,ch,1) = (bs, ch_red, 1) ####CHANGED####
        s=F.sigmoid(torch.matmul(self.w2,intermed))#(1,ch,ch_red * bs, ch_red, 1=bs, ch, 1
        return s*x #bs,ch,seq * bs, ch,1 = bs,ch,seq

# Cell
def weight_init(m):
    '''call weight initialization for model n via n.appy(weight_init)'''
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    if isinstance(m,SqueezeExcite1d):
        stdv1=math.sqrt(2./m.w1.size[0])
        nn.init.normal_(m.w1,0.,stdv1)
        stdv2=math.sqrt(1./m.w2.size[1])
        nn.init.normal_(m.w2,0.,stdv2)

# Cell
def create_head1d(nf, nc, lin_ftrs=None, ps=0.5, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = [ps] if not isinstance(ps,Iterable) else ps
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=False) if act=="relu" else nn.ELU(inplace=False)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.AdaptiveAvgPool1d(1), nn.Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)
    return nn.Sequential(*layers)

# Cell
class basic_conv1d(nn.Sequential):
    '''basic conv1d'''
    def __init__(self, filters=[128,128,128,128],kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1, squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,split_first_layer=False,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        layers = []
        if(isinstance(kernel_size,int)):
            kernel_size = [kernel_size]*len(filters)
        for i in range(len(filters)):
            layers_tmp = []

            layers_tmp.append(_conv1d(input_channels if i==0 else filters[i-1],filters[i],kernel_size=kernel_size[i],stride=(1 if (split_first_layer is True and i==0) else stride),dilation=dilation,act="none" if ((headless is True and i==len(filters)-1) or (split_first_layer is True and i==0)) else act, bn=False if (headless is True and i==len(filters)-1) else bn,drop_p=(0. if i==0 else drop_p)))
            if((split_first_layer is True and i==0)):
                layers_tmp.append(_conv1d(filters[0],filters[0],kernel_size=1,stride=1,act=act, bn=bn,drop_p=0.))
                #layers_tmp.append(nn.Linear(filters[0],filters[0],bias=not(bn)))
                #layers_tmp.append(_fc(filters[0],filters[0],act=act,bn=bn))
            if(pool>0 and i<len(filters)-1):
                layers_tmp.append(nn.MaxPool1d(pool,stride=pool_stride,padding=(pool-1)//2))
            if(squeeze_excite_reduction>0):
                layers_tmp.append(SqueezeExcite1d(filters[i],squeeze_excite_reduction))
            layers.append(nn.Sequential(*layers_tmp))

        #head
        #layers.append(nn.AdaptiveAvgPool1d(1))
        #layers.append(nn.Linear(filters[-1],num_classes))
        #head #inplace=False leads to a runtime error see ReLU+ dropout https://discuss.pytorch.org/t/relu-dropout-inplace/13467/5
        self.headless = headless
        if(headless is True):
            head = nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten())
        else:
            head=create_head1d(filters[-1], nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)

        super().__init__(*layers)

    def get_layer_groups(self):
        return (self[2],self[-1])

    def get_output_layer(self):
        if self.headless is False:
            return self[-1][-1]
        else:
            return None

    def set_output_layer(self,x):
        if self.headless is False:
            self[-1][-1] = x


# Cell
def fcn(filters=[128]*5,num_classes=2,input_channels=8,**kwargs):
    filters_in = filters + [num_classes]
    return basic_conv1d(filters=filters_in,kernel_size=3,stride=1,pool=2,pool_stride=2,input_channels=input_channels,act="relu",bn=True,headless=True)

def fcn_wang(num_classes=2,input_channels=8,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, **kwargs):
    return basic_conv1d(filters=[128,256,128],kernel_size=[8,5,3],stride=1,pool=0,pool_stride=2, num_classes=num_classes,input_channels=input_channels,act="relu",bn=True,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def schirrmeister(num_classes=2,input_channels=8,kernel_size=10,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, **kwargs):
    return basic_conv1d(filters=[25,50,100,200],kernel_size=kernel_size, stride=3, pool=3, pool_stride=1, num_classes=num_classes, input_channels=input_channels, act="relu", bn=True, headless=False,split_first_layer=True,drop_p=0.5,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def sen(filters=[128]*5,num_classes=2,input_channels=8,kernel_size=3,squeeze_excite_reduction=16,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, **kwargs):
    return basic_conv1d(filters=filters,kernel_size=kernel_size,stride=2,pool=0,pool_stride=0,input_channels=input_channels,act="relu",bn=True,num_classes=num_classes,squeeze_excite_reduction=squeeze_excite_reduction,drop_p=drop_p,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def basic1d(filters=[128]*5,kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1, squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, **kwargs):
    return basic_conv1d(filters=filters,kernel_size=kernel_size, stride=stride, dilation=dilation, pool=pool, pool_stride=pool_stride, squeeze_excite_reduction=squeeze_excite_reduction, num_classes=num_classes, input_channels=input_channels, act=act, bn=bn, headless=headless,drop_p=drop_p,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func and hasattr(m, 'weight'): func(m.weight)
    with torch.no_grad():
        if getattr(m, 'bias', None) is not None: m.bias.fill_(0.)
    return m

def _get_norm(prefix, nf, zero=False, **kwargs):
    "Norm layer with `nf` features initialized depending on `norm_type`."
    bn = getattr(nn, f"{prefix}1d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn

def BatchNorm(nf, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, zero=norm_type==NormType.BatchZero, **kwargs)

class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=True, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=nn.ReLU, init=nn.init.kaiming_normal_, xtra=None, **kwargs):
        if padding is None: padding = ((ks-1)//2)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        if bias is None: bias = not(bn)
        conv_func = nn.Conv1d
        conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs), init)
        layers = [conv]
        act_bn = []
        if act_cls is not None: act_bn.append(act_cls())
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)

#adapted from https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py
class MHSA1d(nn.Module):
    def __init__(self, n_dims, length=14, heads=4):
        super(MHSA1d, self).__init__()
        self.heads = heads

        self.query = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv1d(n_dims, n_dims, kernel_size=1)

        self.rel = nn.Parameter(torch.randn([1, heads, n_dims // heads, length]), requires_grad=True)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, length = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1) #128,4,16,16
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1) #128,4,16,16
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, length)

        return out

class ResBlock(nn.Module):
    "Resnet block from `ni` to `nh` with `stride`"
    def __init__(self, expansion, ni, nf, stride=1, kernel_size=3, groups=1, nh1=None, nh2=None, dw=False, g2=1,
                 norm_type=NormType.Batch, act_cls=nn.ReLU, pool=nn.AvgPool1d, pool_first=True, heads=4, mhsa=False, input_size=None, **kwargs):
        super().__init__()
        assert(mhsa is False or expansion>1)
        norm2 = (NormType.BatchZero if norm_type==NormType.Batch else norm_type)
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, **kwargs)
        if(expansion ==1):
            layers  = [ConvLayer(ni,  nh2, kernel_size, stride=stride, groups=ni if dw else groups, **k0),ConvLayer(nh2,  nf, kernel_size, groups=g2, **k1)]
        else:
            layers = [ConvLayer(ni,  nh1, 1, **k0)]
            if(mhsa==False):
                layers.append(ConvLayer(nh1, nh2, kernel_size, stride=stride, groups=nh1 if dw else groups, **k0))
            else:
                assert(nh1==nh2)
                layers.append(MHSA1d(nh1, length=int(input_size), heads=heads))
                if stride == 2:
                    layers.append(nn.AvgPool1d(2, 2))
            layers.append(ConvLayer(nh2,  nf, 1, groups=g2, **k1))
         
        self.convs = nn.Sequential(*layers)
        convpath = [self.convs]
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, **kwargs))
        if stride!=1: idpath.insert((1,0)[pool_first], pool(2, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=False) if act_cls is nn.ReLU else act_cls()

    def forward(self, x): 
        return self.act(self.convpath(x) + self.idpath(x))

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

class XResNet1d(nn.Sequential):
    def __init__(self, block, expansion, layers, input_channels=3, num_classes=1000, stem_szs=(32,32,64), input_size=1000, heads=4, mhsa=False, kernel_size=5,kernel_size_stem=5,
                 widen=1.0, act_cls=nn.ReLU, lin_ftrs_head=None, ps_head=0.5, bn_head=True, act_head="relu", concat_pooling=False, **kwargs):
        self.block = block
        self.expansion = expansion
        self.act_cls = act_cls

        stem_szs = [input_channels, *stem_szs]
        stem = [ConvLayer(stem_szs[i], stem_szs[i+1], ks=kernel_size_stem, stride=2 if i==0 else 1, act_cls=act_cls)
                for i in range(3)]
        stem.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        if(input_size is not None):
            self.input_size = math.floor((input_size-1)/2+1) #strided conv and MaxPool1d
            self.input_size = math.floor((self.input_size-1)/2+1)

        #block_szs = [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
        block_szs = [int(o*widen) for o in [64,64,64,64] +[32]*(len(layers)-4)]
        block_szs = [64//expansion] + block_szs
        #ATTENTION- this uses the S1 Botnet variant with stride 1 in the final block
        blocks = [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                   stride=1 if i==0 else (1 if i==len(layers)-1 and mhsa else 2), kernel_size=kernel_size, heads=heads, mhsa=mhsa if i==len(layers)-1 else False, **kwargs)
                  for i,l in enumerate(layers)]
        
        head = create_head1d(block_szs[-1]*expansion, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)

        super().__init__(
            *stem,
            *blocks,
            head,
        )
        init_cnn(self)

    def _make_layer(self, ni, nf, blocks, stride, kernel_size, heads=4, mhsa=False, **kwargs):
        input_size0 = self.input_size
        input_size1 = math.floor((self.input_size-1)/stride+1) if self.input_size is not None else None
        self.input_size = input_size1
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      kernel_size=kernel_size, act_cls=self.act_cls, heads=heads, mhsa=mhsa, input_size=input_size0 if i==0 else input_size1, **kwargs)
              for i in range(blocks)])

    def get_layer_groups(self):
        return (self[3],self[-1])

    def get_output_layer(self):
        return self[-1][-1]

    def set_output_layer(self,x):
        self[-1][-1]=x

def _xresnet1d(expansion, layers, **kwargs):
    return XResNet1d(ResBlock, expansion, layers, **kwargs)

def xresnet1d18 (**kwargs): return _xresnet1d(1, [2, 2,  2, 2], **kwargs)
def xresnet1d34 (**kwargs): return _xresnet1d(1, [3, 4,  6, 3], **kwargs)
def xresnet1d50 (**kwargs): return _xresnet1d(4, [3, 4,  6, 3], **kwargs)
def xresnet1d101(**kwargs): return _xresnet1d(4, [3, 4, 23, 3], **kwargs)
def xresnet1d152(**kwargs): return _xresnet1d(4, [3, 8, 36, 3], **kwargs)
def xresnet1d18_deep  (**kwargs): return _xresnet1d(1, [2,2,2,2,1,1], **kwargs)
def xresnet1d34_deep  (**kwargs): return _xresnet1d(1, [3,4,6,3,1,1], **kwargs)
def xresnet1d50_deep  (**kwargs): return _xresnet1d(4, [3,4,6,3,1,1], **kwargs)
def xresnet1d18_deeper(**kwargs): return _xresnet1d(1, [2,2,1,1,1,1,1,1], **kwargs)
def xresnet1d34_deeper(**kwargs): return _xresnet1d(1, [3,4,6,3,1,1,1,1], **kwargs)
def xresnet1d50_deeper(**kwargs): return _xresnet1d(4, [3,4,6,3,1,1,1,1], **kwargs)

def xbotnet1d50 (**kwargs): return _xresnet1d(4, [3, 4,  6, 3], mhsa=True, **kwargs)
def xbotnet1d101(**kwargs): return _xresnet1d(4, [3, 4, 23, 3], mhsa=True, **kwargs)
def xbotnet1d152(**kwargs): return _xresnet1d(4, [3, 8, 36, 3], mhsa=True, **kwargs)

class ECGResNet(nn.Module):

    def __init__(self, base_model, out_dim, widen=1.0, big_input=False, concat_pooling=True):
        super(ECGResNet, self).__init__()
        self.resnet_dict = { "xresnet1d50": xresnet1d50(widen=widen, concat_pooling=concat_pooling),
                            "xresnet1d101": xresnet1d101(widen=widen, concat_pooling=concat_pooling)}

        resnet = self._get_basemodel(base_model)
    
        list_of_modules = list(resnet.children())
        
        self.features = nn.Sequential(*list_of_modules[:-1], list_of_modules[-1][0])
        num_ftrs = resnet[-1][-1].in_features
        
        if big_input:
            resnet[0][0] = nn.Conv1d(12, 32, kernel_size=25, stride=10, padding=10)
        else:
            resnet[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm1d(512 if concat_pooling else 256)
        self.drop = nn.Dropout(p=0.5)
        self.l1 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.bn(h)
        x = self.drop(x)
        x = self.l1(x)
        return x