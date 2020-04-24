import os
import time
import cv2
from inference import *
import argparse

def main():

    img_in_path = args.input
    exp_dir=args.trn_dir
    model_type=args.model_type
    test_prototxt=exp_dir + ('/test_rgb_fp16.prototxt' if args.fp16 else '/test_rgb.prototxt')
    test_caffemodel='%s/%s/trn_iter_%d.caffemodel' % (exp_dir, model_type, args.iter)
    extra_pad = args.extra_pad
    if not os.path.isdir(args.outputdir):
      os.makedirs(args.outputdir)
    outputfile = args.outputdir + '/' + args.outputfile

    st = time.time()
    net=load_network(prototxt=test_prototxt, caffemodel=test_caffemodel)
    log.info('DCNN Weights:  %s' % test_caffemodel)
    log.info('DCNN Model Definition:  %s' % test_prototxt)
    log.info('Loading the DCNN network took %0.4f secs\n' % (time.time() - st))

    t = time.time()
    log.info('Reading %s' % img_in_path)
    (img_in, H, W) = prepare_input(img_in_path, fac=8, extra_pad=extra_pad)
    log.info('Input preprocessing took %0.4f secs' % (time.time() - t))
    t = time.time()
    img_out = forward_once(net, img_in, H, W, extra_pad=extra_pad)
    log.info('DCNN forward pass took %0.4f secs (input size: [H = %d, W = %d])' % ((time.time() - t), H, W))
    log.info('Saving output in %s' % outputfile)
    imsave(img_out, out_file=outputfile)
    elapsed=time.time() - st
    log.info('Total Time: %0.4f secs' % elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command for running the Inpainting Pipeline on RGB satellite imagery")
    parser.add_argument('--gpu', type=str, default='0', help='the gpu that will be used, e.g "0"')
    parser.add_argument('--model-type', type=str, default='rgbm_ui', help='Model Type: either "rgbm" or "rgbm_ui"')
    parser.add_argument('--input', type=str, default='./example_input.png', help='path to the 3-channel RGB input file.')
    parser.add_argument('--outputdir', type=str, default='./results', help='path to write output prediction')
    parser.add_argument('--outputfile', type=str, default='example_output.png', help='name of inpainted output RGB')
    parser.add_argument('--fp16', action='store_true', default=False, help='whether to use FP16 inference.')
    parser.add_argument('--trn-dir', type=str, default='./models', help='directory which contains caffe model for inference')
    parser.add_argument('--iter', type=int, default=50000, help='which iteration model to choose (def: 50000)')
    parser.add_argument('--extra-pad', type=int, default=0, help='add extra mirror padding to input '
                                                                 '[sometimes improves results at border pixels] (def: 0)')
    args = parser.parse_args()
    log.info(args)
    main()
