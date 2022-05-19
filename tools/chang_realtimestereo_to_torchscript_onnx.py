#!/usr/bin/env python3

import torch
from torch import Tensor
import torch.onnx

import torch.nn.functional as F

import numpy as np

from pathlib import Path
import sys

import cv2

from torchvision import transforms

rtstereo_dir = sys.argv[1]
rtstereo_model = sys.argv[2]

sys.path.insert(0, rtstereo_dir)
from models import RTStereoNet

def save_torchscript(net, output_file, device):
    scripted_module = torch.jit.script(net)
    # net = net.to(device)
    # sample_input = (torch.zeros(1,3,256,256).to(device), torch.zeros(1,3,256,256).to(device))
    # scripted_module = torch.jit.trace(net, sample_input)
    torch.jit.save(scripted_module, output_file)
    return scripted_module

def save_onnx(net, output_file):
    torch.onnx.export(net,                   # model being run
                  sample_input,              # model input (or a tuple for multiple inputs)
                  output_file,               # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['left', 'right'],   # the model's input names
                  output_names = ['disparity'], # the model's output names
                  dynamic_axes={'left' : {0 : 'batch_size', 2 : 'width', 3 : 'height' },  # variable length axes, except channels
                                'right' : {0 : 'batch_size', 2 : 'width', 3 : 'height' },
                                'output' : {0 : 'batch_size', 2 : 'width', 3 : 'height'}})

def show_color_disparity (name: str, disparity_map: np.ndarray):
    min_disp = 0
    max_disp = 64
    norm_disparity_map = 255*((disparity_map-min_disp) / (max_disp-min_disp))
    disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_MAGMA)
    cv2.imshow (name, disparity_color)

if __name__ == "__main__":
    checkpoint_file = rtstereo_model
    net = RTStereoNet(maxdisp=192, device='cpu')
    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    # Hacky way to check the original model and make sure the export
    # is not screwing up the results.
    if False:
        # left = cv2.imread("datasets/drivingstereo/2018-07-11-14-48-52/2018-07-11-14-48-52_2018-07-11-14-50-10-570_left.jpg", cv2.IMREAD_COLOR)
        # right = cv2.imread("datasets/drivingstereo/2018-07-11-14-48-52/2018-07-11-14-48-52_2018-07-11-14-50-10-570_right.jpg", cv2.IMREAD_COLOR)

        # left = cv2.imread("datasets/eth3d_lowres/forest_2s/im0.png", cv2.IMREAD_COLOR)
        # right = cv2.imread("datasets/eth3d_lowres/forest_2s/im1.png", cv2.IMREAD_COLOR)

        # left = cv2.imread("datasets/eth3d_lowres/playground_3l/im0.png", cv2.IMREAD_COLOR)
        # right = cv2.imread("datasets/eth3d_lowres/playground_3l/im1.png", cv2.IMREAD_COLOR)

        left = cv2.imread("datasets/sceneflow/driving_left.png", cv2.IMREAD_COLOR)
        right = cv2.imread("datasets/sceneflow/driving_right.png", cv2.IMREAD_COLOR)

        # left = cv2.resize (left, (1280,720), cv2.INTER_AREA)
        # right = cv2.resize (right, (1280,720), cv2.INTER_AREA)
        
        imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        img_to_tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**imagenet_stats),
        ])

        left = img_to_tensor_transforms(left)
        right = img_to_tensor_transforms(right)

        # pad to width and hight to 16 times
        if left.shape[1] % 16 != 0:
            times = left.shape[1]//16       
            top_pad = (times+1)*16 -left.shape[1]
        else:
            top_pad = 0

        if left.shape[2] % 16 != 0:
            times = left.shape[2]//16                       
            right_pad = (times+1)*16-left.shape[2]
        else:
            right_pad = 0    

        left = F.pad(left,(0,right_pad, top_pad,0)).unsqueeze(0)
        right = F.pad(right,(0,right_pad, top_pad,0)).unsqueeze(0)

        output = net (left, right)
        output = output[0].detach().numpy().transpose(1,2,0)
        show_color_disparity ("disparity", output)
        cv2.waitKey(0)

    # save_torchscript(net, "chang-realtime-stereo.scripted.pt", torch.device('cpu'))
    # save_torchscript(net, "chang-realtime-stereo-gpu.scripted.pt", torch.device('cuda'))
    
    # Only tracing worked without substantial changes to the codebase.
    device = torch.device('cpu')
    for w,h in [(1280, 720), (640,480), (320,240), (160,128)]:
        sample_input = (torch.zeros(1,3,h,w).to(device), torch.zeros(1,3,h,w).to(device))
        scripted_module = torch.jit.trace(net, sample_input)
        torch.jit.save(scripted_module, f"chang-realtime-stereo-cpu-{w}x{h}.scripted.pt")

        # Need opset16 for grid sampling, currently needs pytorch nightly to do it (1.11 won't).
        # However the exported onnx fails to run:
        # [ONNXRuntimeError] : 1 : FAIL : Load model from chang-realtime-stereo-cpu-320x240.onnx
        #  failed:Type Error: Type parameter (T) of Optype (Mul) bound to different types
        #  (tensor(float) and tensor(int64) in node (Mul_1675).
        if False:
            torch.onnx.export(scripted_module,                 # model being run
                        sample_input,              # model input (or a tuple for multiple inputs)
                        f"chang-realtime-stereo-cpu-{w}x{h}.onnx", # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['left', 'right'],   # the model's input names
                        output_names = ['disparity']) # the model's output names
                        # No dynamic axes with tracing :-(
                        # dynamic_axes={'left' : {0 : 'batch_size', 2 : 'width', 3 : 'height' },  # variable length axes, except channels
                        #             'right' : {0 : 'batch_size', 2 : 'width', 3 : 'height' },
                        #             'output' : {0 : 'batch_size', 2 : 'width', 3 : 'height'}})
    # save_onnx(scripted_module, "chang-realtime-stereo-cpu.onnx")
