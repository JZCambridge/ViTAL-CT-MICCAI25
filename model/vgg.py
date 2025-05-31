"""VGG

Adapted from https://github.com/pytorch/vision 'vgg.py' (BSD-3-Clause) with a few changes for
timm functionality.

Copyright 2021 Ross Wightman
"""
from typing import Union, Callable, List, Optional, Tuple, Union, Dict, Any, cast
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm_3d.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm_3d.layers import ClassifierHead
# from ._builder import build_model_with_cfg
# from ._features_fx import register_notrace_module
# from ._registry import register_model, generate_default_cfgs

__all__ = ['VGG']


cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
_int_tuple_2_t = Union[int, Tuple[int, int, int]]
def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1


def adaptive_avgmax_pool3d(x, output_size: _int_tuple_2_t = 1):
    x_avg = F.adaptive_avg_pool3d(x, output_size)
    x_max = F.adaptive_max_pool3d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool3d(x, output_size: _int_tuple_2_t = 1):
    x_avg = F.adaptive_avg_pool3d(x, output_size)
    x_max = F.adaptive_max_pool3d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool3d(x, pool_type='avg', output_size: _int_tuple_2_t = 1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool3d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool3d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool3d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool3d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x


class FastAdaptiveAvgPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: F = 'NCHWD'):
        super(FastAdaptiveAvgPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.mean(self.dim, keepdim=not self.flatten)


class FastAdaptiveMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHWD'):
        super(FastAdaptiveMaxPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.amax(self.dim, keepdim=not self.flatten)


class FastAdaptiveAvgMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHWD'):
        super(FastAdaptiveAvgMaxPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        x_avg = x.mean(self.dim, keepdim=not self.flatten)
        x_max = x.amax(self.dim, keepdim=not self.flatten)
        return 0.5 * x_avg + 0.5 * x_max


class FastAdaptiveCatAvgMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHWD'):
        super(FastAdaptiveCatAvgMaxPool, self).__init__()
        self.flatten = flatten
        self.dim_reduce = get_spatial_dim(input_fmt)
        if flatten:
            self.dim_cat = 1
        else:
            self.dim_cat = get_channel_dim(input_fmt)

    def forward(self, x):
        x_avg = x.mean(self.dim_reduce, keepdim=not self.flatten)
        x_max = x.amax(self.dim_reduce, keepdim=not self.flatten)
        return torch.cat((x_avg, x_max), self.dim_cat)


class AdaptiveAvgMaxPool3d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super(AdaptiveAvgMaxPool3d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool3d(x, self.output_size)


class AdaptiveCatAvgMaxPool3d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super(AdaptiveCatAvgMaxPool3d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool3d(x, self.output_size)


class SelectAdaptivePool3d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(
            self,
            output_size: _int_tuple_2_t = 1,
            pool_type: str = 'fast',
            flatten: bool = False,
            input_fmt: str = 'NCHWD',
    ):
        super(SelectAdaptivePool3d, self).__init__()
        assert input_fmt in ('NCHWD', 'NHWDC')
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        if not pool_type:
            self.pool = nn.Identity()  # pass through
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        elif pool_type.startswith('fast') or input_fmt != 'NCHWD':
            assert output_size == 1, 'Fast pooling and non NCHWD input formats require output_size == 1.'
            if pool_type.endswith('catavgmax'):
                self.pool = FastAdaptiveCatAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('avgmax'):
                self.pool = FastAdaptiveAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('max'):
                self.pool = FastAdaptiveMaxPool(flatten, input_fmt=input_fmt)
            else:
                self.pool = FastAdaptiveAvgPool(flatten, input_fmt=input_fmt)
            self.flatten = nn.Identity()
        else:
            assert input_fmt == 'NCHWD'
            if pool_type == 'avgmax':
                self.pool = AdaptiveAvgMaxPool3d(output_size)
            elif pool_type == 'catavgmax':
                self.pool = AdaptiveCatAvgMaxPool3d(output_size)
            elif pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool3d(output_size)
            else:
                self.pool = nn.AdaptiveAvgPool3d(output_size)
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x
    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)


class ConvMlp(nn.Module):

    def __init__(
            self,
            in_features=512,
            out_features=4096,
            kernel_size=7,
            mlp_ratio=1.0,
            drop_rate: float = 0.2,
            act_layer: nn.Module = None,
            conv_layer: nn.Module = None,
    ):
        super(ConvMlp, self).__init__()
        self.input_kernel_size = kernel_size
        mid_features = int(out_features * mlp_ratio)
        self.fc1 = conv_layer(in_features, mid_features, kernel_size, bias=True)
        self.act1 = act_layer(True)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = conv_layer(mid_features, out_features, 1, bias=True)
        self.act2 = act_layer(True)

    def forward(self, x):
        if x.shape[-3] < self.input_kernel_size or x.shape[-2] < self.input_kernel_size or x.shape[-1] < self.input_kernel_size:
            # keep the input size >= 7x7
            output_size = (max(self.input_kernel_size, x.shape[-3]), max(self.input_kernel_size, x.shape[-2]), max(self.input_kernel_size, x.shape[-1]))
            x = F.adaptive_avg_pool3d(x, output_size)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class VGG(nn.Module):

    def __init__(
            self,
            cfg: List[Any],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            mlp_ratio: float = 1.0,
            act_layer: nn.Module = nn.ReLU,
            conv_layer: nn.Module = nn.Conv3d,
            norm_layer: nn.Module = nn.BatchNorm3d,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
    ) -> None:
        super(VGG, self).__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.num_features = 4096
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.use_norm = norm_layer is not None
        self.feature_info = []
        prev_chs = in_chans
        net_stride = 1
        pool_layer = nn.MaxPool3d
        layers: List[nn.Module] = []
        for v in cfg:
            last_idx = len(layers) - 1
            if v == 'M':
                self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{last_idx}'))
                layers += [pool_layer(kernel_size=2, stride=2)]
                net_stride *= 2
            else:
                v = cast(int, v)
                conv3d = conv_layer(prev_chs, v, kernel_size=3, padding=1)
                if norm_layer is not None:
                    layers += [conv3d, norm_layer(v), act_layer(inplace=True)]
                else:
                    layers += [conv3d, act_layer(inplace=True)]
                prev_chs = v
        self.features = nn.Sequential(*layers)
        self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{len(layers) - 1}'))

        self.pre_logits = ConvMlp(
            prev_chs,
            self.num_features,
            7,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            act_layer=act_layer,
            conv_layer=conv_layer,
        )
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
        )

        self._initialize_weights()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        # this treats BN layers as separate groups for bn variants, a lot of effort to fix that
        return dict(stem=r'^features\.0', blocks=r'^features\.(\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _filter_fn(state_dict):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        k_r = k
        k_r = k_r.replace('classifier.0', 'pre_logits.fc1')
        k_r = k_r.replace('classifier.3', 'pre_logits.fc2')
        k_r = k_r.replace('classifier.6', 'head.fc')
        if 'classifier.0.weight' in k:
            v = v.reshape(-1, 512, 7, 7)
        if 'classifier.3.weight' in k:
            v = v.reshape(-1, 4096, 1, 1)
        out_dict[k_r] = v
    return out_dict

def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        pretrained_cfg: Optional[Dict] = None,
        pretrained_cfg_overlay: Optional[Dict] = None,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[Dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs,
):
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls: Model class
        variant: Model variant name
        pretrained: Load the pretrained weights
        pretrained_cfg: Model's pretrained weight/task config
        pretrained_cfg_overlay: Entries that will override those in pretrained_cfg
        model_cfg: Model's architecture config
        feature_cfg: Feature extraction adapter config
        pretrained_strict: Load pretrained weights strictly
        pretrained_filter_fn: Filter callable for pretrained weights
        cache_dir: Override model cache dir for Hugging Face Hub and Torch checkpoints
        kwargs_filter: Kwargs keys to filter (remove) before passing to model
        **kwargs: Model args passed through to model __init__
    """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    # pretrained_cfg = resolve_pretrained_cfg(
    #     variant,
    #     pretrained_cfg=pretrained_cfg,
    #     pretrained_cfg_overlay=pretrained_cfg_overlay
    # )
    # pretrained_cfg = pretrained_cfg.to_dict()

    # _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')
        if 'feature_cls' in kwargs:
            feature_cfg['feature_cls'] = kwargs.pop('feature_cls')

    # Instantiate the model
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    if pruned:
        model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        load_pretrained(
            model,
            pretrained_cfg=pretrained_cfg,
            num_classes=num_classes_pretrained,
            in_chans=kwargs.get('in_chans', 3),
            filter_fn=pretrained_filter_fn,
            strict=pretrained_strict,
            cache_dir=cache_dir,
        )

    # Wrap the model in a feature extraction module if enabled
    if features:
        use_getter = False
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()

                # flatten_sequential only valid for some feature extractors
                if feature_cls not in ('dict', 'list', 'hook'):
                    feature_cfg.pop('flatten_sequential', None)

                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                elif feature_cls == 'list':
                    feature_cls = FeatureListNet
                elif feature_cls == 'dict':
                    feature_cls = FeatureDictNet
                elif feature_cls == 'fx':
                    feature_cls = FeatureGraphNet
                elif feature_cls == 'getter':
                    use_getter = True
                    feature_cls = FeatureGetterNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        else:
            feature_cls = FeatureListNet

        output_fmt = getattr(model, 'output_fmt', None)
        if output_fmt is not None and not use_getter:  # don't set default for intermediate feat getter
            feature_cfg.setdefault('output_fmt', output_fmt)

        model = feature_cls(model, **feature_cfg)
        model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)  # add back pretrained cfg
        model.default_cfg = model.pretrained_cfg  # alias for rename backwards compat (default_cfg -> pretrained_cfg)

    return model


def _create_vgg(variant: str, pretrained: bool, **kwargs: Any) -> VGG:
    cfg = variant.split('_')[0]
    # NOTE: VGG is one of few models with stride==1 features w/ 6 out_indices [0..5]
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3, 4, 5))
    model = build_model_with_cfg(
        VGG,
        variant,
        pretrained,
        model_cfg=cfgs[cfg],
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        pretrained_filter_fn=_filter_fn,
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 128, 128, 128), 'pool_size': (4, 4, 4),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.0', 'classifier': 'head.fc',
        **kwargs
    }

def _create_pool(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: Optional[str] = None,
):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool3d(
        pool_type=pool_type,
        flatten=flatten_in_pool,
        input_fmt=input_fmt,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv3d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc

def create_classifier(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: str = 'NCHWD',
        drop_rate: Optional[float] = None,
):
    global_pool, num_pooled_features = _create_pool(
        num_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )
    fc = _create_fc(
        num_pooled_features,
        num_classes,
        use_conv=use_conv,
    )
    if drop_rate is not None:
        dropout = nn.Dropout(drop_rate)
        return global_pool, dropout, fc
    return global_pool, fc

class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            use_conv: bool = False,
            input_fmt: str = 'NCHWD',
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
        """
        super(ClassifierHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        global_pool, fc = create_classifier(
            in_features,
            num_classes,
            pool_type,
            use_conv=use_conv,
            input_fmt=input_fmt,
        )
        self.global_pool = global_pool
        self.drop = nn.Dropout(drop_rate)
        self.fc = fc
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def reset(self, num_classes, pool_type=None):
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                pool_type=pool_type,
                use_conv=self.use_conv,
                input_fmt=self.input_fmt,
            )
            self.flatten = nn.Flatten(1) if self.use_conv and pool_type else nn.Identity()
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features,
                num_classes,
                use_conv=self.use_conv,
            )

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)
# default_cfgs = generate_default_cfgs({
#     'vgg11.tv_in1k': _cfg(hf_hub_id='timm/'),
#     'vgg13.tv_in1k': _cfg(hf_hub_id='timm/'),
#     'vgg16.tv_in1k': _cfg(hf_hub_id='timm/'),
#     'vgg19.tv_in1k': _cfg(hf_hub_id='timm/'),
#     'vgg11_bn.tv_in1k': _cfg(hf_hub_id='timm/'),
#     'vgg13_bn.tv_in1k': _cfg(hf_hub_id='timm/'),
#     'vgg16_bn.tv_in1k': _cfg(hf_hub_id='timm/'),
#     'vgg19_bn.tv_in1k': _cfg(hf_hub_id='timm/'),
# })


def vgg11(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg('vgg11', pretrained=pretrained, **model_args)

def vgg11_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(norm_layer=nn.BatchNorm3d, **kwargs)
    return _create_vgg('vgg11_bn', pretrained=pretrained, **model_args)


def vgg13(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg('vgg13', pretrained=pretrained, **model_args)

def vgg13_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(norm_layer=nn.BatchNorm3d, **kwargs)
    return _create_vgg('vgg13_bn', pretrained=pretrained, **model_args)

def vgg16(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg('vgg16', pretrained=pretrained, **model_args)

def vgg16_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(norm_layer=nn.BatchNorm3d, **kwargs)
    return _create_vgg('vgg16_bn', pretrained=pretrained, **model_args)

def vgg19(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg('vgg19', pretrained=pretrained, **model_args)

def vgg19_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(norm_layer=nn.BatchNorm3d, **kwargs)
    return _create_vgg('vgg19_bn', pretrained=pretrained, **model_args)