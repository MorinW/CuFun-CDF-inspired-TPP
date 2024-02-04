import torch.distributions as td
from CuFun import tpp

import CuFun.tpp.distributions as dist


__all__ = [
    'Exponential',
    'FullyNeuralNet',
    'RMTPP'
    'Const',
    'FullyNeuralNet_Ours',
    'FullyNeuralNet_Ours_Add'
]


def UMNN(config, n_layers=2, layer_size=64, scale_init=1.0, trainable_affine=False, **kwargs):
    base_dist = dist.UMNN(config, n_layers=n_layers, layer_size=layer_size)  # for ours model
    transforms = [
        tpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return tpp.flows.TransformedDistribution(transforms, base_dist)

def Exponential(config, n_components=64, hypernet_hidden_sizes=[64], scale_init=1.0,
                shift_init=0.0, trainable_affine=False, use_sofplus=False, **kwargs):
    base_dist = dist.ExponentialDistribution(config, hypernet_hidden_sizes=hypernet_hidden_sizes)
    transforms = [
        tpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return tpp.flows.TransformedDistribution(transforms, base_dist)


def FullyNeuralNet(config, n_layers=2, layer_size=64, scale_init=1.0, trainable_affine=False, **kwargs):
    base_dist = dist.FullyNN_Ori(config, n_layers=n_layers, layer_size=layer_size)  # for original fullynn model
    transforms = [
        tpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return tpp.flows.TransformedDistribution(transforms, base_dist)


def FullyNeuralNet_Ours(config, n_layers=2, layer_size=64, scale_init=1.0, trainable_affine=False, **kwargs):
    base_dist = dist.FullyNN(config, n_layers=n_layers, layer_size=layer_size)  # for ours model
    transforms = [
        tpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return tpp.flows.TransformedDistribution(transforms, base_dist)


def FullyNeuralNet_Ours_Add(config, n_layers=2, layer_size=64, scale_init=1.0, trainable_affine=False, **kwargs):
    base_dist = dist.FullyNN_Add(config, n_layers=n_layers, layer_size=layer_size)
    transforms = [
        tpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return tpp.flows.TransformedDistribution(transforms, base_dist)


def FullyNeuralNet_Norm(config, n_layers=2, layer_size=64, scale_init=1.0, trainable_affine=False, **kwargs):
    base_dist = dist.FullyNN_Norm(config, n_layers=n_layers, layer_size=layer_size)  # for normalized model
    transforms = [
        tpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return tpp.flows.TransformedDistribution(transforms, base_dist)


def FullyNeuralNet_Final(config, n_layers=2, layer_size=64, scale_init=1.0, trainable_affine=False, **kwargs):
    base_dist = dist.FullyNN_Final(config, n_layers=n_layers, layer_size=layer_size)  # for normalized model
    transforms = [
        tpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return tpp.flows.TransformedDistribution(transforms, base_dist)


def RMTPP(config, scale_init=1.0, trainable_affine=False, hypernet_hidden_sizes=[], **kwargs):
    base_dist = dist.GompertzDistribution(config,
                                          hypernet_hidden_sizes=hypernet_hidden_sizes)
    transforms = [
        tpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return tpp.flows.TransformedDistribution(transforms, base_dist)


def Const(config, n_layers=2, layer_size=64, scale_init=1.0, trainable_affine=False, **kwargs):
    base_dist = dist.Const(config, n_layers=n_layers, layer_size=layer_size)
    transforms = [
        tpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return tpp.flows.TransformedDistribution(transforms, base_dist)

