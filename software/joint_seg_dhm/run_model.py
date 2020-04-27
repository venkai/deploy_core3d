from inference import *
import params

def main():

    # First run JOINT model
    test_model(params)

    # Run model that predicts CLS using a 12-ch RBG+MSI+AGL input.
    params.CURR_MODE = params.CLS_PRED_MODE
    params.CLS_PROTOTXT = './models/rgb_msi_agl_to_cls/test.prototxt'
    params.CLS_CAFFEMODEL = './models/rgb_msi_agl_to_cls/trn_iter_50000.caffemodel'
    test_model(params)

    # Run separate models
    params.CURR_MODE=params.SEP_PRED_MODE
    EXP_TYPES = ['rgb_msi', 'rgb', 'msi']
    for exp_type in EXP_TYPES:
        params.CLS_PROTOTXT = './models/{}_to_cls/test.prototxt'.format(exp_type)
        params.CLS_CAFFEMODEL = './models/{}_to_cls/trn_iter_50000.caffemodel'.format(exp_type)
        params.DEPTH_PROTOTXT = './models/{}_to_agl/test.prototxt'.format(exp_type)
        params.DEPTH_CAFFEMODEL = './models/{}_to_agl/trn_iter_50000.caffemodel'.format(exp_type)
        test_model(params)


if __name__ == "__main__":
    main()
