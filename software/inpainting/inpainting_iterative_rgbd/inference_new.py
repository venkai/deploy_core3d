import caffe
from skimage import color, io
import cv2
import numpy as np
import glog as log
import os
import tifffile
import time

def load_img(imgPath):
    """
    Load image
    :param imgPath: path of the image to load
    :return: numpy array of the image
    """
    log.info('Reading %s' % imgPath)
    if imgPath.endswith('.tif'):
        img = tifffile.imread(imgPath).astype(np.float32)
        if len(img.shape)==3 and img.shape[2] == 3:
            img=img[:,:,::-1]
        # img = cv2.imread(imgPath,-1).astype(np.float32)
    else:
        img = cv2.imread(imgPath)
    if img.ndim == 2:
        img = img[:,:,np.newaxis]
    # if img.shape[2] == 3:
        # img=img[:,:,::-1]
    return img

def load_network(prototxt='models/test.prototxt',
                 caffemodel='models/trn_iter_15000.caffemodel',
                 gpu_num=0):
    caffe.set_device(gpu_num)
    caffe.set_mode_gpu()
    # caffe.set_mode_cpu()
    t=time.time()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    log.info('DCNN Weights:  %s' % caffemodel)
    log.info('DCNN Model Definition:  %s' % prototxt)
    log.info('Loading the DCNN network took %0.4f secs\n' % (time.time() - t))
    return net

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
    return (img_in, H, W)

# fac = 2 ^ (number of times image is downsampled in the network)
def prepare_input_old(img_in, fac=8, extra_pad=0, pad_mode='reflect', read_img=True):
    if read_img:
        img_in = cv2.imread(img_in)
    if img_in.ndim == 2:
        # force color conversion
        img_in = np.repeat(img_in[:,:,np.newaxis],3,axis=2)
    # Pad H,W to make them multiples of fac
    H=img_in.shape[0]
    W=img_in.shape[1]
    padH = (fac - (H % fac)) % fac
    padW = (fac - (W % fac)) % fac
    if padH>0 or padW>0 or extra_pad>0:
        img_in = np.pad(img_in,pad_width=((extra_pad,padH+extra_pad),(extra_pad,padW+extra_pad),(0,0)),mode=pad_mode)
    img_in=img_in.astype(np.float32)
    return (img_in, H, W)

def transform_img(img, ind, inv_transform=False):
    if inv_transform and ind == 5:
       ind=6
    elif inv_transform and ind == 6:
        ind=5
    if ind == 0:
        # No Transformation
        return img
    elif ind == 1:
        # Horizontal Flip
        img = img[:,::-1,:]
    elif ind == 2:
        # Vertical Flip
        img = img[::-1,...]
    elif ind == 3:
        # Horizontal Flip + Vertical Flip
        img = img[::-1,::-1,:]
    elif ind == 4:
        # Transpose
        img = np.transpose(img,[1,0,2])
    elif ind == 5:
        # Horizontal Flip + Transpose
        img = np.transpose(img[:,::-1,:],[1,0,2])
    elif ind == 6:
        # Vertical Flip + Transpose
        img = np.transpose(img[::-1,...],[1,0,2])
    elif ind == 7:
        # Horizontal Flip + Vertical Flip + Transpose
        img = np.transpose(img[::-1,::-1,:],[1,0,2])
    return img

def forward(net, img_in, H, W, extra_pad=0):

    img_out = np.zeros(shape=(H,W,3,8), dtype=np.float32)
    for ind in range(8):
        img_in_curr = transform_img(img_in, ind)
        # H*W*C to C*H*W
        img_in_curr=np.transpose(img_in_curr,[2,0,1])
        # Add Batch Dimension (C*H*W to 1*C*H*W)
        img_in_curr = img_in_curr[np.newaxis,:,:,:]
        net.blobs['data'].reshape(*img_in_curr.shape)
        # net.reshape()
        net.blobs['data'].data[...] = img_in_curr
        net.forward()
        img_out_curr = net.blobs['pred'].data[0,...].transpose([1,2,0])
        img_out_curr=np.clip(img_out_curr,0,255)
        img_out_curr = transform_img(img_out_curr, ind, inv_transform=True)
        img_out[..., ind] = img_out_curr[extra_pad:H+extra_pad, extra_pad:W+extra_pad, :]

    img_out = np.mean(img_out, axis=3)
    img_out=img_out.astype(np.uint8)
    return img_out

def forward_once(net, img_in, H, W, extra_pad=0):
    # H*W*C to C*H*W
    img_in=np.transpose(img_in,[2,0,1])
    # Add Batch Dimension (C*H*W to 1*C*H*W)
    img_in = img_in[np.newaxis,:,:,:]
    net.blobs['data'].reshape(*img_in.shape)
    net.reshape()
    net.blobs['data'].data[...] = img_in
    log.info('Input shape: {}'.format(img_in.shape))
    t=time.time()
    net.forward()
    log.info('DCNN forward pass took %0.4f secs (input size: [H = %d, W = %d])' % ((time.time() - t), H, W))
    img_out = net.blobs['res2_00'].data[0,...].transpose([1,2,0])
    img_out = img_out[extra_pad:H+extra_pad, extra_pad:W+extra_pad, :]
    return img_out

def forward_tiled(prototxt, caffemodel, img_in, H, W, out_ch):
    img_out_shape = (img_in.shape[0], img_in.shape[1], out_ch)
    img_out = np.zeros(img_out_shape, np.float32)
    # H*W*C to C*H*W
    img_in=np.transpose(img_in,[2,0,1])
    # Add Batch Dimension (C*H*W to 1*C*H*W)
    img_in = img_in[np.newaxis,:,:,:]
    overlap = 32; N = 1408
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
            net=load_network(prototxt=prototxt, caffemodel=caffemodel)
            img_in_curr = img_in[:,:,lhi:uhi,lwi:uwi]
            net.blobs['data'].reshape(*img_in_curr.shape)
            net.reshape()
            net.blobs['data'].data[...] = img_in_curr
            log.info('Input shape: {}'.format(img_in_curr.shape))
            t=time.time()
            net.forward()
            log.info('DCNN forward pass took %0.4f secs' % (time.time() - t))
            img_out_curr = net.blobs['res2_00'].data[0,...].transpose([1,2,0])
            log.info('IMG_OUT_CURR.shape: {}'.format(img_out_curr.shape))
            log.info('[CTR: %d] img_out[%d:%d, %d:%d, :] = img_out_curr[%d:%d, %d:%d, :]'% (
                ctr, lho, uho, lwo, uwo, lho - lhi, uho - lhi, lwo - lwi, uwo - lwi
            ))
            img_out[lho:uho, lwo:uwo, :] = img_out_curr[lho - lhi:uho - lhi, lwo - lwi:uwo - lwi, :]
            ctr += 1
            del net
    img_out = img_out[0:H, 0:W, :]
    return img_out



    


def imsave(img, out_file='./out.png'):
    log.info('Saving {} output in {}'.format(img.shape, out_file))
    dtype=np.uint8 if img.ndim==3 else np.float32
    if out_file.endswith('.tif'):
        if img.ndim==3:
            img=np.clip(img[:,:,::-1],0,255)
        tifffile.imsave(out_file, img.astype(dtype), compress=6)
    else:
        img = np.clip(img,0,255)
        cv2.imwrite(out_file, img.astype(dtype))

  
def imsave_old(img, out_file='./out.png'):
    cv2.imwrite(out_file, img)

