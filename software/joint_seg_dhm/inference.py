import caffe

import numpy as np
import cv2
from glob import glob
import os
import tifffile
import glog as log
import time

def get_image_paths(params):
    return dict(
        rgbPaths = sorted(glob(os.path.join(params.TEST_DIR, 'RGB/*%s*.%s' % (params.IMG_FILE_STR,params.IMG_FILE_EXT)))),
        msiPaths = sorted(glob(os.path.join(params.TEST_DIR, 'MSI/*%s*.%s' % (params.MSI_FILE_STR,params.IMG_FILE_EXT)))),
        dhmPaths = sorted(glob(os.path.join(params.TEST_DIR, 'AGL/*%s*.%s' % (params.DEPTH_FILE_STR,params.IMG_FILE_EXT)))),
    )


def load_img(imgPath):
    """
    Load image
    :param imgPath: path of the image to load
    :return: numpy array of the image
    """
    log.info('Reading %s' % imgPath)
    img = tifffile.imread(imgPath).astype(np.float32)
    if img.ndim == 2:
        img = img[:,:,np.newaxis]
    if img.shape[2] == 3:
        img=img[:,:,::-1] # RGB to BGR
    return img

def split_img_based_on_exp(img_in_curr, exp_type='rgb_msi'):
    img_in = img_in_curr.copy()
    if exp_type == 'rgb_msi_agl':
        return img_in
    elif exp_type == 'rgb_msi':
        return img_in[:,:,:-1]
    elif exp_type == 'rgb':
        return img_in[:,:,0:3]
    else:
        return img_in[:,:,3:-1]


def convert_labels(Lorig, params, toLasStandard=True):
    """
    Convert the labels from the original CLS file to consecutive integer values starting at 0
    :param Lorig: numpy array containing original labels
    :param params: input parameters from params.py
    :param toLasStandard: determines if labels are converted from the las standard labels to training labels
        or from training labels to the las standard
    :return: Numpy array containing the converted labels
    """
    L = Lorig.copy()
    if toLasStandard:
        labelMapping = params.LABEL_MAPPING_TRAIN2LAS
    else:
        labelMapping = params.LABEL_MAPPING_LAS2TRAIN

    for key,val in labelMapping.items():
        L[Lorig==key] = val

    return L

def prepare_mult_channel_img(fnames,basename='',axis=2):
    res = load_img(basename+fnames[0])
    for i in range(1,len(fnames)):
        I = load_img(basename+fnames[i])
        res = np.concatenate((res,I),axis=axis)
    return res

def get_prototxt_caffemodel(params,mode):
    if mode == params.CLS_PRED_MODE:
        prototxt = params.CLS_PROTOTXT
        caffemodel = params.CLS_CAFFEMODEL
    elif mode == params.DEPTH_PRED_MODE:
        prototxt = params.DEPTH_PROTOTXT
        caffemodel = params.DEPTH_CAFFEMODEL
    else:
        prototxt = params.PROTOTXT
        caffemodel = params.CAFFEMODEL
    return [prototxt, caffemodel]

def deploy_model(img_in, params, mode=2):
    st = time.time()
    H, W, _ = img_in.shape
    [prototxt, caffemodel] = get_prototxt_caffemodel(params,mode)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    log.info('DCNN Weights:  %s' % caffemodel)
    log.info('DCNN Model Definition:  %s' % prototxt)
    t=time.time()
    log.info('Loading the DCNN network took %0.4f secs\n' % (t - st))
    # H*W*C to C*H*W
    img_in_curr = img_in.copy()
    img_in_curr=np.transpose(img_in_curr,[2,0,1])
    # Add Batch Dimension (C*H*W to 1*C*H*W)
    img_in_curr = img_in_curr[np.newaxis,:,:,:]
    log.info('Input Shape: {}'.format(img_in_curr.shape))
    if not H==320 or not W==320:
        net.blobs['data'].reshape(*img_in.shape)
        net.reshape()
    net.blobs['data'].data[...] = img_in_curr
    net.forward()
    img_out = net.blobs['pred'].data[0,...].transpose([1,2,0]).copy()
    log.info('Output Shape: {}'.format(img_out.shape))
    log.info('DCNN forward pass took %0.4f secs' % (time.time() - t))
    del net
    return img_out


def test_joint_model(params):
    """
    Joint model for simultaneously predicting DHM and CLS
    :return: None
    """
    caffe.set_mode_gpu()
    caffe.set_device(params.GPU_NUM)
    OUTPUT_DIR = os.path.join(params.OUTPUT_DIR,'joint')
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    allPaths = get_image_paths(params)
    rgbPaths = allPaths['rgbPaths']
    msiPaths = allPaths['msiPaths']

    Nimg = len(rgbPaths)
    log.info('Number of files = %d' % Nimg)
    st=time.time()
    for ind in range(Nimg):
        imageName = os.path.split(rgbPaths[ind])[-1]
        dhmOutName = imageName.replace(params.IMG_FILE_STR, params.DEPTH_FILE_STR)
        clsOutName = imageName.replace(params.IMG_FILE_STR, params.CLASS_FILE_STR)
        out_cls_file = os.path.join(OUTPUT_DIR, clsOutName)
        out_dhm_file = os.path.join(OUTPUT_DIR, dhmOutName)
        if os.path.isfile(out_dhm_file) and os.path.isfile(out_cls_file):
                log.info('Files Already Exist: %s\n Files already exist: %s', out_cls_file, out_dhm_file)
                continue
        img_in = prepare_mult_channel_img([rgbPaths[ind],msiPaths[ind]])
        img_out = deploy_model(img_in, params)
        pred_dhm = img_out[:,:,0]
        pred_cls = img_out[:,:,1:]
        if params.NUM_CATEGORIES > 1:
            pred_cls = np.argmax(pred_cls, axis=2).astype('uint8')
        else:
            pred_cls = (pred_cls > params.BINARY_CONF_TH).astype('uint8')
        if params.CONVERT_LABELS:
            pred_cls = convert_labels(pred_cls, params, toLasStandard=True)

        log.info('[Image %d/%d] Saving predicted DHM to %s' % (ind + 1, Nimg, out_dhm_file))
        tifffile.imsave(out_dhm_file, pred_dhm, compress=6)
        log.info('[Image %d/%d] Saving predicted CLS to %s\n' % (ind + 1, Nimg, out_cls_file))
        tifffile.imsave(out_cls_file, pred_cls, compress=6)
        log.info('[Image %d/%d] Total Time Elapsed = %0.4f secs\n\n' % (ind + 1, Nimg, time.time() - st))
        time.sleep(0.2)

def test_model(params):
    """
    Launches separate models for DHM prediction and CLS prediction
    :return: None
    """
    if params.CURR_MODE == params.JOINT_PRED_MODE:
        test_joint_model(params)
        return
    caffe.set_mode_gpu()
    caffe.set_device(params.GPU_NUM)
    exp_cls_type = params.CLS_CAFFEMODEL.split('/')[-2].split('_to')[0]
    OUTPUT_CLS_DIR = os.path.join(params.OUTPUT_DIR,exp_cls_type)
    if not os.path.isdir(OUTPUT_CLS_DIR):
        os.makedirs(OUTPUT_CLS_DIR)
    exp_dhm_type = params.DEPTH_CAFFEMODEL.split('/')[-2].split('_to')[0]
    OUTPUT_DHM_DIR = os.path.join(params.OUTPUT_DIR,exp_dhm_type)
    if not os.path.isdir(OUTPUT_DHM_DIR):
        os.makedirs(OUTPUT_DHM_DIR)
    allPaths = get_image_paths(params)
    rgbPaths = allPaths['rgbPaths']
    msiPaths = allPaths['msiPaths']
    dhmPaths = allPaths['dhmPaths']
    Nimg = len(rgbPaths)
    log.info('Number of files = %d' % Nimg)
    st=time.time()
    for ind in range(Nimg):
        imageName = os.path.split(rgbPaths[ind])[-1]
        dhmOutName = imageName.replace(params.IMG_FILE_STR, params.DEPTH_FILE_STR)
        clsOutName = imageName.replace(params.IMG_FILE_STR, params.CLASS_FILE_STR)
        out_cls_file=os.path.join(OUTPUT_CLS_DIR, clsOutName)
        out_dhm_file=os.path.join(OUTPUT_DHM_DIR, dhmOutName)
        if os.path.isfile(out_cls_file) and os.path.isfile(out_dhm_file):
            log.info('File Already Exists: %s', out_cls_file)
            log.info('File Already Exists: %s', out_dhm_file)
            continue
        img_in = prepare_mult_channel_img([rgbPaths[ind],msiPaths[ind],dhmPaths[ind]])
        if params.CURR_MODE == params.CLS_PRED_MODE or params.CURR_MODE == params.SEP_PRED_MODE and not os.path.isfile(out_cls_file):
            img_in_curr = split_img_based_on_exp(img_in, exp_type=exp_cls_type)
            pred_cls = deploy_model(img_in_curr, params, mode=params.CLS_PRED_MODE)
            if params.NUM_CATEGORIES > 1:
                pred_cls = np.argmax(pred_cls, axis=2).astype('uint8')
            else:
                pred_cls = (pred_cls > params.BINARY_CONF_TH).astype('uint8')
            if params.CONVERT_LABELS:
                pred_cls = convert_labels(pred_cls, params, toLasStandard=True)
            log.info('[Image %d/%d] Saving predicted CLS to %s' % (ind + 1, Nimg, out_cls_file))
            tifffile.imsave(out_cls_file, pred_cls, compress=6)
        if params.CURR_MODE == params.DEPTH_PRED_MODE or params.CURR_MODE == params.SEP_PRED_MODE and not os.path.isfile(out_dhm_file):
            img_in_curr = split_img_based_on_exp(img_in, exp_type=exp_dhm_type)
            pred_dhm = deploy_model(img_in_curr, params, mode=params.DEPTH_PRED_MODE)
            log.info('[Image %d/%d] Saving predicted DHM to %s' % (ind + 1, Nimg, out_dhm_file))
            tifffile.imsave(out_dhm_file, pred_dhm, compress=6)
        log.info('[Image %d/%d] Total Time Elapsed = %0.4f secs' % (ind + 1, Nimg, time.time() - st))



