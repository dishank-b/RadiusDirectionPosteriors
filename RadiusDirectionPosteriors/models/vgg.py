import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import BayesianModel
from torch_user.nn.bayesian_modules.radial_conv import RadialConv2d
from torch_user.nn.bayesian_modules.radial_linear import RadialLinear
from models.vgg_prior import VGG16Prior

class VGG16(BayesianModel):
    def __init__(self, prior_info):
        super(VGG16, self).__init__()

        conv_channel_base = 32
        in_channels = 3
        self.prior = VGG16Prior(prior_info, conv_channel_base)
        b1c1, b1c2, b2c1, b2c2, b3c1, b3c2, b3c3, b4c1, b4c2, b4c3, b5c1, b5c2, b5c3, fc = self.prior()

        self.activation = torch.nn.LeakyReLU(0.2)
        self.block_1_conv_1 = RadialConv2d(in_channels, conv_channel_base, kernel_size=3, prior=b1c1, padding=1, stride=2, with_global=True)
        self.block_1_conv_2 = RadialConv2d(conv_channel_base, conv_channel_base, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_2_conv_1 = RadialConv2d(conv_channel_base, conv_channel_base * 2, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_2_conv_2 = RadialConv2d(conv_channel_base * 2, conv_channel_base * 2, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_3_conv_1 = RadialConv2d(conv_channel_base * 2, conv_channel_base * 4, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_3_conv_2 = RadialConv2d(conv_channel_base * 4, conv_channel_base * 4, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_3_conv_3 = RadialConv2d(conv_channel_base * 4, conv_channel_base * 4, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_4_conv_1 = RadialConv2d(conv_channel_base * 4, conv_channel_base * 8, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_4_conv_2 = RadialConv2d(conv_channel_base * 8, conv_channel_base * 8, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_4_conv_3 = RadialConv2d(conv_channel_base * 8, conv_channel_base * 8, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_5_conv_1 = RadialConv2d(conv_channel_base * 8, conv_channel_base * 8, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_5_conv_2 = RadialConv2d(conv_channel_base * 8, conv_channel_base * 8, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)
        self.block_5_conv_3 = RadialConv2d(conv_channel_base * 8, conv_channel_base * 8, kernel_size=3, prior=b1c1, padding=1, stride=1, with_global=True)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.global_mean = nn.AdaptiveMaxPool2d((1,1))
        self.global_max = nn.AdaptiveAvgPool2d((1,1))

        self.linear = RadialLinear(in_features=conv_channel_base * 16, out_features=10, bias=True, prior=fc, with_global=True)

    def forward(self, x):
        # Input has shape [examples, samples, channels, height, width]

        x = self.block_1_conv_1(x)
        x = self.activation(x)
        x = self.block_1_conv_2(x)
        x = self.activation(x)

        x = self.max_pool(x)

        x = self.block_2_conv_1(x)
        x = self.activation(x)
        x = self.block_2_conv_2(x)
        x = self.activation(x)

        x = self.max_pool(x)

        x = self.block_3_conv_1(x)
        x = self.activation(x)
        x = self.block_3_conv_2(x)
        x = self.activation(x)
        x = self.block_3_conv_3(x)
        x = self.activation(x)

        x = self.max_pool(x)

        x = self.block_4_conv_1(x)
        x = self.activation(x)
        x = self.block_4_conv_2(x)
        x = self.activation(x)
        x = self.block_4_conv_3(x)
        x = self.activation(x)

        x = self.max_pool(x)

        x = self.block_5_conv_1(x)
        x = self.activation(x)
        x = self.block_5_conv_2(x)
        x = self.activation(x)
        x = self.block_5_conv_3(x)
        x = self.activation(x)

        x_1 = self.global_mean(x)
        x_2 = self.global_max(x)

        x = torch.cat((x_1, x_2), dim=1)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x

    def kl_divergence(self):
        kld = self.block_1_conv_1.kl_divergence()
        kld += self.block_1_conv_2.kl_divergence()
        kld += self.block_2_conv_1.kl_divergence()
        kld += self.block_2_conv_2.kl_divergence()
        kld += self.block_3_conv_1.kl_divergence()
        kld += self.block_3_conv_2.kl_divergence()
        kld += self.block_3_conv_3.kl_divergence()
        kld += self.block_4_conv_1.kl_divergence()
        kld += self.block_4_conv_2.kl_divergence()
        kld += self.block_4_conv_3.kl_divergence()
        kld += self.block_5_conv_1.kl_divergence()
        kld += self.block_5_conv_2.kl_divergence()
        kld += self.block_5_conv_3.kl_divergence()
        kld += self.linear.kl_divergence()
        return kld

    def deterministic_forward(self, set_deterministic):
        for m in self.modules():
            if hasattr(m, 'deterministic_forward') and m != self:
                m.deterministic_forward(set_deterministic)

    def init_hyperparam_value(model):
        hyperparam_info_list = []
        for name, m in model.named_modules():
            if hasattr(m, 'init_hyperparams_repr'):
                hyperparam_info_list.append('%s(%s) : %s' % (name, m._get_name(), m.init_hyperparams_repr()))
        return '\n'.join(hyperparam_info_list)
