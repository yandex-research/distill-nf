import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.RRDBNet_arch import RRDBNet
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet
from utils.util import opt_get


class SRFlowNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None, opt=None, step=None):
        super(SRFlowNet, self).__init__()

        self.opt = opt
        self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
                            None else opt_get(opt, ['datasets', 'train', 'quant'])
        self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'hidden_channels'])
        hidden_channels = hidden_channels or 64
        # for p in self.RRDB.parameters():
        #     p.requires_grad = False
        self.RRDB_training = True  # Default is true
        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
    
        self.flowUpsamplerNet = FlowUpsamplerNet((160, 160, 3), hidden_channels, K,
                                                flow_coupling=opt['network_G']['flow']['coupling'], opt=opt)
    
    def forward(self, gt=None, lr=None, z=None, eps_std=None, epses=None, reverse_with_grad=False, lr_enc=None):
        if reverse_with_grad:
            return self.reverse_flow(lr, z, eps_std=eps_std, epses=epses, lr_enc=lr_enc)
        else:
            with torch.no_grad():
                return self.reverse_flow(lr, z, eps_std=eps_std, epses=epses, lr_enc=lr_enc)

    def reverse_flow(self, lr, z, eps_std, epses=None, lr_enc=None):
        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)
        x = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, epses=epses)
        return x

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr, get_steps=True)
        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        if len(block_idxs) > 0:
            concat = torch.cat([rrdbResults["block_{}".format(idx)] for idx in block_idxs], dim=1)

            if opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'concat']) or False:
                keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
                if 'fea_up0' in rrdbResults.keys():
                    keys.append('fea_up0')
                if 'fea_up-1' in rrdbResults.keys():
                    keys.append('fea_up-1')
                if self.opt['scale'] >= 8:
                    keys.append('fea_up8')
                if self.opt['scale'] == 16:
                    keys.append('fea_up16')
                for k in keys:
                    h = rrdbResults[k].shape[2]
                    w = rrdbResults[k].shape[3]
                    rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, (h, w))], dim=1)
        return rrdbResults