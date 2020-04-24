from __future__ import division

import os
import time
from glob import glob

import cv2
import glog as log
import numpy as np
import tifffile

import caffe


def get_input_paths(params):
    return sorted(glob(os.path.join(params.INPUT_DIR, '%s*%s*.%s' % (
        '*/' * params.SUBFOLDER_DEPTH, params.MSI_FILE_STR, params.IMG_FILE_EXT))))


def get_output_paths(input_paths, params, out_str):
    input_dirs, img_names = zip(*[os.path.split(p) for p in input_paths])
    output_dirs = [d.replace(params.INPUT_DIR, params.OUTPUT_DIR) for d in input_dirs]
    out_names = [p.replace(
        params.MSI_FILE_STR + '.' + params.IMG_FILE_EXT,
        out_str + '.' + params.OUT_FILE_EXT) for p in img_names]
    output_paths = [os.path.join(d, p) for d, p in zip(output_dirs, out_names)]
    if not params.MK_OUTPUT_DIRS:
        return output_paths
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    return output_paths


def load_img(img_path):
    """
    Load image
    :param imgPath: path of the image to load
    :return: numpy array of the image
    """
    log.info('Reading %s' % img_path)
    img = tifffile.imread(img_path).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.shape[2] == 3:
        img = img[:, :, ::-1]  # RGB to BGR
    return img


# fac = 2 ^ (number of times image is downsampled in the network)
def prepare_input(img_in, fac=8, extra_pad=0, pad_mode='reflect', read_img=True):
    if read_img:
       img_in = load_img(img_in)
    # Pad H,W to make them multiples of fac
    H=img_in.shape[0]
    W=img_in.shape[1]
    padH = (fac - (H % fac)) % fac
    padW = (fac - (W % fac)) % fac
    if padH>0 or padW>0 or extra_pad>0:
        img_in = np.pad(img_in,pad_width=((extra_pad,padH+extra_pad),(extra_pad,padW+extra_pad),(0,0)),mode=pad_mode)
    img_in=img_in.astype(np.float32)
    log.info(img_in.shape)
    img_in = normalize_msi(img_in)
    return (img_in, H, W)


def normalize_msi(img_in):
    log.info('Normalizing input MSI')
    for i in range(img_in.shape[2]):
        mu = np.mean(img_in[:,:,i])
        std = np.std(img_in[:,:,i], ddof=1)
        log.info('Channel #%d: mean = %0.2f, st-dev = %0.2f'% (i + 1, mu, std))
        img_in[:,:,i] -= mu
        img_in[:,:,i] /= std
    return img_in



def deploy_model(img_in, params):
    st = time.time()
    net = caffe.Net(params.PROTOTXT, params.CAFFEMODEL, caffe.TEST)
    net.params['bn_data'][0].data[...]=0
    net.params['bn_data'][1].data[...]=1
    log.info('DCNN Weights:  %s' % params.CAFFEMODEL)
    log.info('DCNN Model Definition:  %s' % params.PROTOTXT)
    t = time.time()
    log.info('Loading the DCNN network took %0.4f secs\n' % (t - st))
    # H*W*C to C*H*W
    img_in = np.transpose(img_in, [2, 0, 1])
    # Add Batch Dimension (C*H*W to 1*C*H*W)
    img_in = img_in[np.newaxis, :, :, :]
    log.info('Input Shape: {}'.format(img_in.shape))
    net.blobs['data'].data[...] = img_in
    net.forward()
    img_out = net.blobs['pred'].data[0, ...].transpose([1, 2, 0]).copy()
    log.info('Output Shape: {}'.format(img_out.shape))
    log.info('DCNN forward pass took %0.4f secs' % (time.time() - t))
    del net
    return img_out

def deploy_model_tiled(img_in, H, W, params):
    out_ch = params.OUT_CH[params.EXP_TYPE]
    st = time.time()
    log.info('DCNN Weights:  %s' % params.CAFFEMODEL)
    log.info('DCNN Model Definition:  %s' % params.PROTOTXT)
    t = time.time()
    log.info('Loading the DCNN network took %0.4f secs\n' % (t - st))
    # H*W*C to C*H*W
    img_out_shape = (img_in.shape[0], img_in.shape[1], out_ch)
    img_out = np.zeros(img_out_shape, np.float32)
    # H*W*C to C*H*W
    img_in=np.transpose(img_in,[2,0,1])
    # Add Batch Dimension (C*H*W to 1*C*H*W)
    img_in = img_in[np.newaxis,:,:,:]
    overlap = 128; N = 1024
    Htot = img_in.shape[2]; Wtot = img_in.shape[3]
    ctr=1
    for h in range(0, Htot, N):
        lhi = max(0, h - overlap)
        uhi = min(Htot, h + N + overlap)
        lho = h
        uho = min(Htot, h + N)
        for w in range(0, Wtot, N):
            lwi = max(0, w - overlap)
            uwi = min(Wtot, w + N + overlap)
            lwo = w
            uwo = min(Wtot, w + N)
            net = caffe.Net(params.PROTOTXT, params.CAFFEMODEL, caffe.TEST)
            net.params['bn_data'][0].data[...]=0
            net.params['bn_data'][1].data[...]=1
            img_in_curr = img_in[:,:,lhi:uhi,lwi:uwi]
            net.blobs['data'].reshape(*img_in_curr.shape)
            net.reshape()
            net.blobs['data'].data[...] = img_in_curr
            log.info('Input shape: {}'.format(img_in_curr.shape))
            t=time.time()
            net.forward()
            log.info('DCNN forward pass took %0.4f secs' % (time.time() - t))
            img_out_curr = net.blobs['pred'].data[0,...].transpose([1,2,0])
            log.info('IMG_OUT_CURR.shape: {}'.format(img_out_curr.shape))
            log.info('[CTR: %d] img_out[%d:%d, %d:%d, :] = img_out_curr[%d:%d, %d:%d, :]'% (
                ctr, lho, uho, lwo, uwo, lho - lhi, uho - lhi, lwo - lwi, uwo - lwi
            ))
            img_out[lho:uho, lwo:uwo, :] = img_out_curr[lho - lhi:uho - lhi, lwo - lwi:uwo - lwi, :]
            ctr += 1
            del net
    img_out = img_out[0: H, 0: W, :]
    img_out = np.clip(img_out, 0, 255)
    return img_out



def test_model(params):
    """
    Launches model for converting MSI to either:
    1) RGB                 [if params.EXP_TYPE == msi_to_rgb]
    2) RGB + SRGB          [if params.EXP_TYPE == msi_to_srgb]
    3) RGB + SRGB + SMASK  [if params.EXP_TYPE == msi_to_smask]
    :return: None
    """
    caffe.set_mode_gpu()
    caffe.set_device(params.GPU_NUM)
    input_paths = get_input_paths(params)
    for I in input_paths:
        log.info(I)
    out_rgb_paths = get_output_paths(input_paths, params, params.RGB_FILE_STR)
    params.MK_OUTPUT_DIRS = False
    out_srgb_paths = get_output_paths(input_paths, params, params.SRGB_FILE_STR)
    out_smask_paths = get_output_paths(input_paths, params, params.SMASK_FILE_STR)
    
    Nimg = len(input_paths)
    log.info('Number of files = %d' % Nimg)
    st = time.time()
    for ind in range(Nimg):
        img_in, H, W = prepare_input(input_paths[ind])
        pred = deploy_model_tiled(img_in, H, W, params)
        
        log.info('[Image %d/%d] Saving RGB to %s' % (ind + 1, Nimg, out_rgb_paths[ind]))
        pred_curr = pred[:,:,0:3]
        pred_curr = np.clip(pred_curr[:,:,::-1], 0, 255).astype(np.uint8)
        tifffile.imsave(out_rgb_paths[ind], pred_curr, compress=6)
        if pred.shape[2] == 3:
            continue
        
        log.info('[Image %d/%d] Saving Shadow-Free RGB to %s' % (ind + 1, Nimg, out_srgb_paths[ind]))
        pred_curr = pred[:,:,3:6]
        pred_curr = np.clip(pred_curr[:,:,::-1], 0, 255).astype(np.uint8)
        tifffile.imsave(out_srgb_paths[ind], pred_curr, compress=6)
        if pred.shape[2] == 6:
            continue
        
        log.info('[Image %d/%d] Saving Shadow-Logits to %s' % (ind + 1, Nimg, out_smask_paths[ind]))
        pred_curr = pred[:,:,-1]
        tifffile.imsave(out_smask_paths[ind], pred_curr, compress=6)
        
        log.info('[Image %d/%d] Total Time Elapsed = %0.4f secs' %
                 (ind + 1, Nimg, time.time() - st))
        