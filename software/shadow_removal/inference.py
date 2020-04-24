import caffe
from skimage import color, io
import cv2
import numpy as np
import glog as log
import os
import tifffile
import time

def load_network(prototxt='models/test.prototxt',
                 caffemodel='models/trn_iter_15000.caffemodel',
                 gpu_num=0):
  caffe.set_mode_gpu()
  caffe.set_device(gpu_num)
  log.info('DCNN Weights:  %s' % caffemodel)
  log.info('DCNN Model Definition:  %s' % prototxt)
  return caffe.Net(prototxt, caffemodel, caffe.TEST)

# fac = 2 ^ (number of times image is downsampled in the network)
def prepare_input(img_in, fac=8, extra_pad=0, pad_mode='reflect', read_img=True):
  if read_img:
    log.info('Reading %s' % img_in)
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

# Forward Pass
def forward_once(net, img_in, H, W):
  t=time.time()
  # H*W*C to C*H*W
  img_in=np.transpose(img_in,[2,0,1])
  # Add Batch Dimension (C*H*W to 1*C*H*W)
  img_in = img_in[np.newaxis,:,:,:]
  log.info('Input Shape: {}'.format(img_in.shape))
  if not H==1024 or not W==1024:
    net.blobs['data'].reshape(*img_in.shape)
    net.reshape()
  net.blobs['data'].data[...] = img_in
  net.forward()
  img_out = net.blobs['res2_00'].data[0,...].transpose([1,2,0]).copy()
  img_out = img_out[0:H, 0:W, :]
  log.info('Output Shape: {}'.format(img_out.shape))
  log.info('DCNN forward pass took %0.4f secs' % (time.time() - t))
  return img_out

# Save Image as tif or png
def imsave(img, out_file='./out.png'):
    log.info('Saving {} output in {}'.format(img.shape, out_file))
    if out_file.endswith('.tif'):
      if img.ndim==3:
        img=img[:,:,::-1]
      tifffile.imsave(out_file, img, compress=6)
    else:
        cv2.imwrite(out_file, img)

