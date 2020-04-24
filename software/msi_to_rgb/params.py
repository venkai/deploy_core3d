import os

# ----------------------------------------------------
# MODEL TESTING
GPU_NUM = 0
US_FP16 = False
EXP_TYPE = 'msi_to_smask'

# ----------------------------------------------------
# DATA I/O

INPUT_DIR = './input_msi'
OUTPUT_DIR = './results'
SUBFOLDER_DEPTH = 0  # recursion depth for subfolders (0 means root, i.e. INPUT_DIR)
MK_OUTPUT_DIRS = True

RGB_FILE_STR = '_RGB'
SRGB_FILE_STR = '_SF'
SMASK_FILE_STR = '_SMASK'
MSI_FILE_STR = ''
IMG_FILE_EXT = 'tif'
OUT_FILE_EXT = 'tif'

# ----------------------------------------------------

OUT_CH = {
    'msi_to_rgb' : 3,
    'msi_to_srgb':  6,
    'msi_to_smask':  7,
}

if MK_OUTPUT_DIRS and not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PROTOTXT = './models/%s/test%s.prototxt' % (EXP_TYPE, '_fp16' if US_FP16 else '')
CAFFEMODEL = './models/%s/trn_iter_50000.caffemodel' % EXP_TYPE
