import caffe
from skimage import color, io
import cv2
import numpy as np
import glog as log
import os


def load_network(prototxt='models/test.prototxt',
                 caffemodel='models/trn_iter_15000.caffemodel',
                 gpu_num=0):
  caffe.set_mode_gpu()
  caffe.set_device(gpu_num)
  # caffe.set_mode_cpu()
  return caffe.Net(prototxt, caffemodel, caffe.TEST)

# fac = 2 ^ (number of times image is downsampled in the network)
def prepare_input(img_in, fac=8, extra_pad=0, pad_mode='reflect', read_img=True):
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

def forward_once(net, img_in, H, W, extra_pad=0, S=1):
  H *= S
  W *= S
  extra_pad *= S
  # H*W*C to C*H*W
  img_in=np.transpose(img_in,[2,0,1])
  # Add Batch Dimension (C*H*W to 1*C*H*W)
  img_in = img_in[np.newaxis,:,:,:]
  net.blobs['data'].reshape(*img_in.shape)
  net.reshape()
  net.blobs['data'].data[...] = img_in
  net.forward()
  img_out = net.blobs['res2_00'].data[0,...].transpose([1,2,0])
  img_out=np.clip(img_out,0,255)
  img_out = img_out[extra_pad:H+extra_pad, extra_pad:W+extra_pad, :]
  img_out=img_out.astype(np.uint8)
  # img_out=cv2.cvtColor(img_out, cv2.COLOR_YCrCb2BGR)
  return img_out


def imsave(img, out_file='./out.png'):
  cv2.imwrite(out_file, img)

