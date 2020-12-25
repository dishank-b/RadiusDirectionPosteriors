from models.utils import double_prior_summary, prior_summary
from torch_user.nn.utils import ml_kappa

PRIOR_EPSILON = 0.1

class VGG16Prior(object):

    def __init__(self, prior_info, conv_channel_base):
        prior_type, prior_hyper = prior_info
        self.b1c1 = None
        self.b1c2 = None
        self.b2c1 = None
        self.b2c2 = None
        self.b3c1 = None
        self.b3c2 = None
        self.b3c3 = None
        self.b4c1 = None
        self.b4c2 = None
        self.b4c3 = None
        self.b5c1 = None
        self.b5c2 = None
        self.b5c3 = None
        self.fc = None

        self._prior_halfcauchy(prior_type, prior_hyper)
        self.b1c1['direction'] = ('vMF', {'concentration': ml_kappa(dim=3 * 9, eps=PRIOR_EPSILON)})
        self.b1c2['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 9, eps=PRIOR_EPSILON)})
        self.b2c1['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 9, eps=PRIOR_EPSILON)})
        self.b2c2['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 2 * 9, eps=PRIOR_EPSILON)})
        self.b3c1['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 2 * 9, eps=PRIOR_EPSILON)})
        self.b3c2['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 4 * 9, eps=PRIOR_EPSILON)})
        self.b3c3['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 4 * 9, eps=PRIOR_EPSILON)})
        self.b4c1['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 4 * 9, eps=PRIOR_EPSILON)})
        self.b4c2['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 8 * 9, eps=PRIOR_EPSILON)})
        self.b4c3['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 8 * 9, eps=PRIOR_EPSILON)})
        self.b5c1['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 8 * 9, eps=PRIOR_EPSILON)})
        self.b5c2['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 8 * 9, eps=PRIOR_EPSILON)})
        self.b5c3['direction'] = ('vMF', {'concentration': ml_kappa(dim=conv_channel_base * 8 * 9, eps=PRIOR_EPSILON)})
        self.fc['direction'] = ('vMF', {'concentration': ml_kappa(dim=10, eps=PRIOR_EPSILON)})

    def __call__(self):
        return self.b1c1, self.b1c2, self.b2c1, self.b2c2, self.b3c1, self.b3c2, self.b3c3, self.b4c1, \
                self.b4c2, self.b4c3, self.b5c1, self.b5c2, self.b5c3, self.fc

    def __repr__(self):
        prior_info_str_list = ['***PRIORS***']
        prior_info_str_list.append('B1C1 : ' + prior_summary(self.b1c1))
        prior_info_str_list.append('B1C2 : ' + prior_summary(self.b1c2))
        prior_info_str_list.append('B2C1 : ' + prior_summary(self.b2c1))
        prior_info_str_list.append('B2C2 : ' + prior_summary(self.b2c2))
        prior_info_str_list.append('B3C1 : ' + prior_summary(self.b3c1))
        prior_info_str_list.append('B3C2 : ' + prior_summary(self.b3c2))
        prior_info_str_list.append('B3C3 : ' + prior_summary(self.b3c3))
        prior_info_str_list.append('B4C1 : ' + prior_summary(self.b4c1))
        prior_info_str_list.append('B4C2 : ' + prior_summary(self.b4c2))
        prior_info_str_list.append('B4C3 : ' + prior_summary(self.b4c3))
        prior_info_str_list.append('B5C1 : ' + prior_summary(self.b5c1))
        prior_info_str_list.append('B5C2 : ' + prior_summary(self.b5c2))
        prior_info_str_list.append('B5C3 : ' + prior_summary(self.b5c3))
        prior_info_str_list.append('FC : ' + prior_summary(self.fc))
        return '\n'.join(prior_info_str_list)

    def _prior_halfcauchy(self, prior_type, prior_hyper):
        tau_fc_local = prior_hyper['tau_fc_local']
        tau_fc_global = prior_hyper['tau_fc_global']
        tau_conv_local = prior_hyper['tau_conv_local']
        tau_conv_global = prior_hyper['tau_conv_global']
        self.b1c1 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b1c2 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b2c1 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b2c2 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b3c1 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b3c2 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b3c3 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b4c1 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b4c2 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b4c3 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b5c1 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b5c2 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.b5c3 = {'radius': (prior_type, {'tau_global': tau_conv_global, 'tau_local': tau_conv_local})}
        self.fc = {'radius': (prior_type, {'tau_global': tau_fc_global, 'tau_local': tau_fc_local})}
