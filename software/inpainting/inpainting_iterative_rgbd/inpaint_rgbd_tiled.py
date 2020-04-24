import os
import time
import cv2
from inference_new import *
import argparse
from glob import glob


def main():
    st = time.time()
    outputdir = args.outputdir
    if not os.path.isdir(outputdir):
      os.makedirs(outputdir)
    
    # Load network
    test_prototxt=args.trn_dir + '/test_' + args.model_type + ('_fp16.prototxt' if args.fp16 else '.prototxt')
    if args.iter > 0:
        test_caffemodel='%s/%s/trn_iter_%d.caffemodel' % (args.trn_dir, args.model_type, args.iter)
    else:
        # Get most recent model
        list_caffemodels=glob('%s/%s/trn_iter_*.caffemodel' % (args.trn_dir, args.model_type))
        test_caffemodel = max(list_caffemodels, key=os.path.getctime)
    
    # Load inputs
    (img_in_rgb, H, W) = prepare_input(args.input_rgb, fac=8, extra_pad=args.extra_pad)
    (img_in_dsm, H, W) = prepare_input(args.input_dsm, fac=8, extra_pad=args.extra_pad)
    (img_in_dtm, H, W) = prepare_input(args.input_dtm, fac=8, extra_pad=args.extra_pad)
    img_in_dhm = img_in_dsm - img_in_dtm
    img_in_dhm[img_in_dhm < 0] = 0
    img_in = np.concatenate((img_in_rgb,img_in_dhm),axis=2)
    
    # Forward pass
    for rp_ind in range(args.nrep):
        log.info('Forward Pass %d/%d' % (rp_ind + 1, args.nrep))
        img_out = forward_tiled(test_prototxt, test_caffemodel, img_in, H, W, out_ch = 4)
        od_gt_id = (img_out[:,:,-1] > img_in[0:H,0:W,-1])
        img_out[:,:,-1][od_gt_id] = img_in[0:H,0:W,-1][od_gt_id]
        img_in[0:H,0:W,-1]  = img_out[:,:,-1]
        
    
    # Save outputs
    img_rgb_out = img_out[:,:,0:3]
    imsave(img_rgb_out, out_file=outputdir + '/' + args.outputfile + '_RGB.tif')
    img_dhm_out = img_out[:,:,-1]
    img_dsm_out = img_dhm_out + img_in_dtm[0:H, 0:W,0]
    imsave(img_dsm_out, out_file=outputdir + '/' + args.outputfile + '_DSM.tif')
    imsave(img_dhm_out, out_file=outputdir + '/' + args.outputfile + '_DHM.tif')
    log.info('Total Time: %0.4f secs' % (time.time() - st))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command for running the Inpainting Pipeline on RGBD satellite imagery")
    parser.add_argument('--gpu', type=str, default='0', help='the gpu that will be used, e.g "0"')
    parser.add_argument('--nrep', type=int, default=9, help='repeated depth-inpainting iterations (def: 9)')
    parser.add_argument('--input-rgb', type=str, default='./example_input_RGB.png', help='path to the 3-channel RGB input file.')
    parser.add_argument('--input-dsm', type=str, default='./example_input_DSM.tif', help='path to the DSM input file.')
    parser.add_argument('--input-dtm', type=str, default='./example_input_DTM.tif', help='path to the DTM input file.')
    parser.add_argument('--outputdir', type=str, default='./results', help='path to write output prediction')
    parser.add_argument('--outputfile', type=str, default='example_output', help='Inpainted output')
    parser.add_argument('--fp16', action='store_true', default=False, help='whether to use FP16 inference.')
    parser.add_argument('--trn-dir', type=str, default='./models', help='directory which contains caffe model for inference')
    parser.add_argument('--iter', type=int, default=0, help='which iteration model to choose (def: 0 [choose latest])')
    parser.add_argument('--model-type', type=str, default='rgbd', help='Model Type')
    parser.add_argument('--extra-pad', type=int, default=0, help='add extra mirror padding to input '
                                                                 '[sometimes improves results at border pixels] (def: 0)')
    args = parser.parse_args()
    log.info(args)
    main()