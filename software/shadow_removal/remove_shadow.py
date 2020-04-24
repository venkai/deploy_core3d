import os
import time
import cv2
from inference import *
import argparse

def main():

    img_in_path = args.input
    exp_dir=args.trn_dir
    test_prototxt=exp_dir + ('/test_fp16.prototxt' if args.fp16 else '/test.prototxt')
    test_caffemodel='%s/trn_iter_%d.caffemodel' % (exp_dir, args.iter)  
    if not os.path.isdir(args.outputdir):
      os.makedirs(args.outputdir)
    st = time.time()
    net=load_network(prototxt=test_prototxt, caffemodel=test_caffemodel)
    log.info('Loading the DCNN network took %0.4f secs\n' % (time.time() - st))
    (img_in, H, W) = prepare_input(img_in_path, fac=8)
    t = time.time()
    img_out = forward_once(net, img_in, H, W)
    img_rgb_out = np.clip(img_out[:,:,0:3],0,255).astype('uint8')
    imsave(img_rgb_out, out_file=args.outputdir + '/' + args.outputrgbfile)
    img_conf_out = img_out[:,:,4] - img_out[:,:,3] - 0.4
    imsave(img_conf_out, out_file=args.outputdir + '/' + args.outputconffile)
    elapsed=time.time() - st
    log.info('Total Time: %0.4f secs' % elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command for running the Shadow Removal Pipeline")
    parser.add_argument('--gpu', type=str, default='0', help='the gpu that will be used, e.g "0"')
    parser.add_argument('--input', type=str, default='./example_input.png', help='path to the 3-channel RGB input file.')
    parser.add_argument('--outputdir', type=str, default='./results', help='path to write output prediction.')
    parser.add_argument('--outputrgbfile', type=str, default='shadow_free_output.png', help='name of shadow-free output RGB')
    parser.add_argument('--outputconffile', type=str, default='shadow_logits.tif', help='shadow confidence values (as logits)')
    parser.add_argument('--fp16', action='store_true', default=False, help='whether to use FP16 inference.')
    parser.add_argument('--trn-dir', type=str, default='./models', help='directory which contains caffe model for inference')
    parser.add_argument('--iter', type=int, default=45000, help='which iteration model to choose (def: 45000)')
    args = parser.parse_args()
    log.info(args)
    main()
