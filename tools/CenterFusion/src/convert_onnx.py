'''
Script to convert a trained CenterNet model to ONNX .
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
from torch.onnx.symbolic_registry import register_op

import copy

from model.model import create_model, load_model
from opts import opts
from dataset.dataset_factory import dataset_factory
from detector import Detector

from lib.utils.ddd_utils import get_pc_hm
from lib.utils.pointcloud import generate_pc_hm

# add onnx symbol `Atan` for torch.atan 
def atan_symbolic(g, input):
    g.op("Atan",input)
register_op("atan", atan_symbolic, 'add onnx symbol for torch.atan ', 9)

def get_binrot_alpha(rot, channel_first=False):
    
  # output: (...,B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[..., 0]
  assert (len(rot) == 4, "Tensor rot need to have 4 dims")
  if isinstance(rot, torch.Tensor):
    if channel_first:
      tan1 = torch.clamp(rot[:,2:3,:,:]/rot[:,3:4,:,:],min=-1e6, max=1e6)
      tan2 = torch.clamp(rot[:,6:7,:,:]/rot[:,7:8,:,:],min=-1e6, max=1e6)

    else:
      tan1 = torch.clamp(rot[..., 2:3]/rot[..., 3:4],min=-1e6, max=1e6)
      tan2 = torch.clamp(rot[..., 6:7]/rot[..., 7:8],min=-1e6, max=1e6)

    alpha1 = torch.atan(tan1) + (-0.5 * np.pi)
    alpha2 = torch.atan(tan2) + ( 0.5 * np.pi)
  # elif isinstance(rot, np.ndarray):
  #   alpha1 = np.arctan2(rot[..., 2], rot[..., 3]) + (-0.5 * np.pi)
  #   alpha2 = np.arctan2(rot[..., 6], rot[..., 7]) + ( 0.5 * np.pi)
  else:
      raise TypeError("Tensor rot dtype is invalid ! ")
  if channel_first:
    idx = rot[:,1:2,:,:] > rot[:,5:6,:,:]
  else:
    idx = rot[..., 1:2] > rot[..., 5:6]
  idx = idx.int()
  alpha = alpha1 * idx + alpha2 * (1-idx)
  alpha[alpha<-np.pi] += 2* np.pi
  alpha[alpha>np.pi] -= 2*np.pi
  return alpha


class ImgModel(torch.nn.Module):
    def __init__(self, net):
        super(ImgModel, self).__init__()
        self.net = net
        self.opt = self.net.opt

    def forward(self, x):
        feats = self.net.img2feats(x)
        out = []
        for s in range(self.net.num_stacks):
            z = {}
            z['feat'] = feats[s]
            ## Run the first stage heads
            for head in self.net.heads:
                if head not in self.net.secondary_heads:
                    z[head] = self.net.__getattr__(head)(feats[s])

            keys= list(z.keys())            
            for head in keys:
                if head == "hm":
                    value = z.pop(head)
                    value = value.sigmoid_()
                    score, pred_label = torch.max(value,dim=1)
                    z["score"] = score
                    z['label'] = pred_label
                elif "dep" in head: # dep or dep_sec
                    z[head] = 1. / (z[head].sigmoid() + 1e-6) - 1.
            out.append(z)
        return out



class FusionModel(torch.nn.Module):
    def __init__(self, net):
        super(FusionModel, self).__init__()
        self.net = net
        self.opt = self.net.opt

    def forward(self, feats,  pc_dep):
        out = []
        pc_hm=pc_dep
        for s in range(self.net.num_stacks):
            z = {}
            sec_feats = [feats[s], pc_hm]
            sec_feats = torch.cat(sec_feats, 1)
            for head in self.net.secondary_heads:
                z[head] = self.net.__getattr__(head)(sec_feats)

            keys= list(z.keys())            
            for head in keys:
                if head == "hm":
                    value = z.pop(head)
                    value = value.sigmoid_()
                    score, pred_label = torch.max(value,dim=1)
                    z["score"] = score
                    z['label'] = pred_label
                elif "dep" in head: # dep or dep_sec
                    z[head] = 1. / (z[head].sigmoid() + 1e-6) - 1.
                # elif "rot" in head:
                #     z[head] = get_binrot_alpha(z[head])

        out.append(z)
        return out

def convert_onnx(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.model_output_list = True
    Dataset = dataset_factory[opt.test_dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.device = torch.device("cuda")
    print(opt)
    model = create_model(
      opt.arch, opt.heads, opt.head_conv, opt=opt)
    model = load_model(model, opt.load_model, opt)
    model = model.to(opt.device)
    img_model = ImgModel(model)
    fusion_model = FusionModel(model)
    img_model.eval()
    fusion_model.eval()
    
    inputs = [x for x in torch.load("../data/cf_inputs.pth")]
    for i in range(len(inputs)):
        inputs[i] = inputs[i].to(opt.device)

    outs1 = img_model(inputs[0])
    feats = [out['feat'] for out in outs1 ]
    outs2   = fusion_model(feats, inputs[1])
    
    torch.onnx.export(
        img_model, (inputs[0]) , 
        "../models/cf_img.onnx", input_names = ("img",),output_names=tuple(outs1[0].keys()), opset_version=11 )
    torch.onnx.export(
        fusion_model, (feats, inputs[1]) , 
        "../models/cf_fus.onnx", input_names = ("feat","pc_dep"),output_names=tuple(outs2[0].keys()), opset_version=11 )
    print("Finished !")
#   dummy_input1 = torch.randn(1, 3, opt.input_h, opt.input_w).to(opt.device)

#   if opt.tracking:
#     dummy_input2 = torch.randn(1, 3, opt.input_h, opt.input_w).to(opt.device)
#     if opt.pre_hm:
#       dummy_input3 = torch.randn(1, 1, opt.input_h, opt.input_w).to(opt.device)
#       torch.onnx.export(
#         model, (dummy_input1, dummy_input2, dummy_input3), 
#         "../models/{}.onnx".format(opt.exp_id))
#     else:
#       torch.onnx.export(
#         model, (dummy_input1, dummy_input2), 
#         "../models/{}.onnx".format(opt.exp_id))
#   else:
#     torch.onnx.export(
#       model, (dummy_input1, ), 
#       "../models/{}.onnx".format(opt.exp_id))
if __name__ == '__main__':
  opt = opts().parse()
  convert_onnx(opt)

