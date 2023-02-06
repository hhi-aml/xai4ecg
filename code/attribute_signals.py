from zennit.attribution import Gradient, IntegratedGradients
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from zennit.core import Composite
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from captum.attr import GuidedGradCam

from zennit.core import Composite
from zennit.layer import Sum
from zennit.rules import Gamma, Epsilon, ZBox, ZPlus, AlphaBeta, Flat, Pass, Norm, ReLUDeconvNet, ReLUGuidedBackprop
from zennit.types import Convolution, Linear, AvgPool, Activation, BatchNorm
from zennit.composites import layer_map_base, LayerMapComposite

class EpsilonComposite(LayerMapComposite):
    '''An explicit composite using the epsilon rule for all layers.

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, epsilon=1e-3, stabilizer=1e-6, layer_map=None, zero_params=None, canonizers=None):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, Epsilon(epsilon=epsilon, **rule_kwargs)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon, **rule_kwargs)),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)


def attribute_signals(model, samples, output_indices, num_classes, modeltype, explain_method, batch_size=16):

    if explain_method != 'gradcam':
        if modeltype == 'lenet':
            canonizer = VGGCanonizer()
        elif modeltype in ['resnet', 'xresnet']:
            canonizer = ResNetCanonizer()

        if explain_method == 'lrp':
            composite = EpsilonComposite(canonizers=[canonizer])
            att = Gradient(model=model, composite=composite)
        elif explain_method == 'gradient':
            att = Gradient(model=model, composite=None)
        elif explain_method == 'ig':
            composite = Composite(canonizers=[canonizer])
            att = IntegratedGradients(model=model, composite=composite, n_iter=10)

        samples = torch.from_numpy(np.swapaxes(samples,1,2)).type(torch.float).to('cuda')
        
        output_matrix = np.zeros((len(output_indices),num_classes))
        for i,idx in enumerate(output_indices):
            output_matrix[i,idx]=1
        output_matrix = torch.from_numpy(output_matrix).to('cuda')

        with att as attributor:
            relevance = []
            for i in range((len(samples)//batch_size)+1):
                _, tmp = attributor(samples[i*batch_size:(i+1)*batch_size], output_matrix[i*batch_size:(i+1)*batch_size])
                tmp=tmp.cpu().detach().numpy()
                relevance.append(tmp)
            relevance = np.concatenate(relevance)
        return np.swapaxes(relevance,1,2)
    else:
        if modeltype == 'resnet':
            feature_model = nn.Sequential(*list(model.model.children()))
            guided_gc = GuidedGradCam(model.model, model.model.layer4)
        elif modeltype=='xresnet':
            guided_gc = GuidedGradCam(model.model, model.model.features)
        elif modeltype == 'lenet':
            guided_gc = GuidedGradCam(model.sequential, model.sequential[-7])

        relevance = []
        for i in tqdm(range((len(samples)//batch_size)+1)):
            tmp = guided_gc.attribute(torch.from_numpy(np.swapaxes(samples[i*batch_size:(i+1)*batch_size],1,2)).type(torch.float).to('cuda'), int(output_indices[0]))#, attribute_to_layer_input=True)
            tmp=tmp.cpu().detach().numpy()
            relevance.append(tmp)
        relevance = np.concatenate(relevance)
        return np.swapaxes(relevance,1,2)
