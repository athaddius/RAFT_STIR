
import sys
sys.path.append('core')

import onnxruntime as ort
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

NUMITERS = 12



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def testconvertmodel(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        imfile1, imfile2 = images[0], images[1]
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=NUMITERS, test_mode=True)
        kwargs = {"iters":NUMITERS, "test_mode": True}
        args = (image1, image2)
        onnx_model = torch.onnx.export(model,
                (image1, image2,kwargs),
                "raftsmall.onnx",
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names = ['image1', 'image2'],
                output_names = ['flow_low', 'flow_up'],
                verbose=False)
        ort_session = ort.InferenceSession("raftsmall.onnx")



        outputs = ort_session.run(
                    None,
                    {"image1": to_numpy(image1), "image2": to_numpy(image2)},
                        )
        estimated_flow = torch.from_numpy(outputs[1]).to(DEVICE)
        if not torch.allclose(flow_up, estimated_flow, atol=1e-2, rtol=1e-2):
            print("ONNX model not close to true model")
            breakpoint()
            print(estimated_flow-flow_up)
        #viz(image1, flow_up)

def convertmodeldirect(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        image1 = torch.randn(1, 3, 512, 640, device="cuda")
        image2 = torch.randn(1, 3, 512, 640, device="cuda") ## THESE MUST be divisible by 8, otherwise need to pad using inputpadder

        NUMITERS=12
        flow_low, flow_up = model(image1, image2, iters=NUMITERS, test_mode=True)
        kwargs = {"iters":NUMITERS, "test_mode": True}
        args = (image1, image2)
        onnx_model = torch.onnx.export(model,
                (image1, image2,kwargs),
                "raftsmall_STIR.onnx",
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names = ['image1', 'image2'],
                output_names = ['flow_low', 'flow_up'],
                verbose=False)


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class RaftPointTrack(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pointlist, image1, image2):
        """ takes in pointlist of size 1 npts 2 (xy)
        im1 1 C H W
        im2 1 C H W
        returns:
        pointlist non-normalized of size npts 2 xy"""
        _, flow_up = self.model(image1, image2, iters=NUMITERS, test_mode=True)

        #pointlist = pointlist.unsqueeze(2) # 1 npts 1 2
        flow_locs = bilinear_sampler(flow_up, pointlist.unsqueeze(2))
        flow_locs = flow_locs.squeeze(3).permute(0,2,1) # N npts 2
        end_locs = pointlist + flow_locs
        return end_locs

def convertmodelpointtrack(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    pointtrack = RaftPointTrack(model).to(DEVICE).eval()

    with torch.no_grad():
        image1 = torch.randn(1, 3, 512, 640, device="cuda")
        image2 = torch.randn(1, 3, 512, 640, device="cuda") ## THESE MUST be divisible by 8, otherwise need to pad using inputpadder
        points = torch.rand(1, 64, 2, device="cuda") * 512. 
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        TEST = False
        if TEST:
            images = sorted(images)
            imfile1, imfile2 = images[0], images[1]
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

        end_locs = pointtrack(points, image1, image2)
        args = (points, image1, image2)
        onnx_model = torch.onnx.export(pointtrack,
                args,
                "raft_pointtrackSTIR.onnx",
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names = ['pointlist', 'image1', 'image2'],
                output_names = ['end_points'],
                dynamic_axes={ "pointlist": {1: "numpts"},
                    "end_points": [1],
                    },
                verbose=False)
        ort_session = ort.InferenceSession("raft_pointtrackSTIR.onnx")



        outputs = ort_session.run(
                    None,
                    {"pointlist": to_numpy(points), "image1": to_numpy(image1), "image2": to_numpy(image2)},
                        )
        estimated_endpoints = torch.from_numpy(outputs[0]).to(DEVICE)
        if TEST:
            if not torch.allclose(end_locs, estimated_endpoints, atol=1e-2, rtol=1e-2):
                print("ONNX model not close to true model")
                breakpoint()
                print(estimated_endpoints-end_locs)


# https://pytorch.org/docs/stable/onnx_torchscript.html
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    args.path = 'demo-frames'
    args.model =  'models/raft-small.pth'
    args.small = True

    testconvertmodel(args)
    convertmodeldirect(args)
    convertmodelpointtrack(args)
