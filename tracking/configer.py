import os
import time

import numpy as np
import tensorflow as tf

import vgg19_tf
from funs_tracking import *
from Logger import *
from vgg_utis import vgg_process_images, vgg_resize_maps

#########################
##### gpu parameter #####
#########################
gpu_id = '/gpu:0'
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

#########################
#### data params ########
#########################

# data_path = r'/data/cyy/Data/CVPR2013Bechmark_Color/'
# cache_path = r'/data/cyy/Results/vgg_cf_all_scale'
data_path = r'/data3/TB-100/OTB2015'
cache_path = r'./Results'
if not os.path.isdir(cache_path):
    os.mkdir(cache_path)
pstr = 'gcnn'

####
padding = {'large': 1, 'height': 0.4, 'generic': 2}  # 25~50: 2.5 others 2.2
cell_size0 = 4
batch_size = 1  # fixed
max_win2 = 1600
min_win2 = 1600
fea_sz = np.asarray([57, 57])
#########################
####### VGG Model #######
#########################

vgg_model_path = '/data2/wangchaoqun/model/vgg19.npy'
vgg_batch_size = 1
vgg_out_layers = np.asarray((10, 11, 12, 14, 15, 16))


vgg_is_lrn = False

# image processing params for vgg
img_param = {}
img_param['interp_tool'] = 'misc'  # misc or skimage
img_param['interp'] = 'bilinear'
img_param['normal_hw'] = (224, 224)
img_param['normal_type'] = 'keep_all_content'

##################################
###### graph parameters ##########
##################################

gh1 = {'height_width': None, 'number_edges': 4, 'metric': 'euclidean', 'normalized_laplacian': True}

# pca params
pca_is_mean = True
pca_is_norm = False
pca_energy = 100
####
nn_p = 6  #
nn_K = 20
nn_gamma = 1.0

####################### cf params ###############################
search_scale = fun_get_search_scale()

kernel_sigma = 0.5
kernel_type = 'linear'
kernel_gamma = 1  # 1.e-6
update_factor = 0.0075  # Jogging learning 0.005, others 0.015
cf_nframe_update = 1
weight_update_factor = 0.01
