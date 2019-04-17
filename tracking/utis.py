'''
some utilities.
-- modified by Cui Zhen, April 18, 2016
'''
# utis.py
############ function list ##############
# build_w_b(rng,n_input,n_out,str_type,factor=1.)
# build_w_b_kernel(rng,kernel,str_type,factor=1.)
# create_dir(dir_name,folder = None)
# elewise_mlp_list(a,b)
# elewise_div_list(a,b)
# fun_ctr2rct(ctrs)
# fun_rct2ctr(rcts)
# infer_pred_shape(gt_lmks,ref_point,ref_width)
# get_convpool_out_shape(inshape,kernel,stride,poolsize)
# get_conv_out_shape(inshape,kernel,stride)
# get_layers_lr_wgt_mul(netparams)
# get_net_layerNo(netparams,keyname)
# get_patch_warping(img,ctr,tmp_sz,win_sz)
# get_path(dir_name, folder)
# get_tuple_from_tuple(x,idx)
# kmeans_cluster(x,k,isDataNorm,isCtrNorm)
# locations_of_substring(string, substring)
# load_vars6_dumps(filepath,n)
# load_variable_list(filepath)
# matrix2string(x,datatype)
# net_params_parsing(netparams)
# net_params_print(netparams,logger)
# pca_te(x, w, x_mean = None, x_norm = None)
# pca_tr(x, energy = 0.9, is_mean = True, is_norm = True)
# read_filenames(fpath,filetype)
# read_image(fpath,isColor,isPad)
# resize_batch(x,nm_h,nm_w,interp_type)
# save_mat_file(filename,data1,data2,data3,data4)
# save_vars6_dumps(filepath,x1,x2,x3,x4,x5,x6)
# save_variable_list(x,filepath,is_shared)
# split_fpaths(fpathfiles)
# swap_columns(x,i1,i2)
# tf_process_images_for_vgg(ims,normal_height,normal_width,interp = 'bilinear')
# vector2string(x,datatype):

#########################################

import glob
import os
import pdb
import pickle

import numpy as np
import scipy.cluster as cluster
import scipy.io
import scipy.misc as MISC
from PIL import Image


def pca_tr(x, energy=0.9, is_mean=True, is_norm=True):
    # x: n * d
    z = np.transpose(x)  # d*n

    if is_mean:
        x_mean = np.mean(z, axis=1, keepdims=True)
        x_mean = np.asarray(x_mean, dtype=x.dtype)
        z = z - x_mean
        x_mean = np.transpose(x_mean)
    else:
        x_mean = 0

    if is_norm:
        x_norm = np.linalg.norm(z, axis=1, keepdims=True)
        x_norm = np.asarray(x_norm, dtype=x.dtype)
        idx = np.where(x_norm < 1.0e-6)
        x_norm[idx] = 1.
        z = z / x_norm
        x_norm = np.transpose(x_norm)
    else:
        x_norm = 1

    #
    d, n = z.shape
    # pdb.set_trace()
    if d >= n:
        ztz = np.dot(np.transpose(z), z)
        a, v = np.linalg.eig(ztz)
        ind = np.argsort(a)[::-1]
        ev = a[ind]
        v = v[:, ind]

        if energy <= 1:
            r = np.cumsum(ev) / np.sum(ev)
            ind = np.where(r >= energy)
            dim = ind[0]
        else:
            dim = energy

        a = ev[0:dim]
        a = np.diag(1 / np.sqrt(a))
        v = v[:, 0:dim]
        w = np.dot(np.dot(z, v), a)
    else:
        zzt = np.dot(z, np.transpose(z))
        a, v = np.linalg.eig(zzt)
        ind = np.argsort(a)[::-1]
        ev = a[ind]
        v = v[:, ind]

        if energy <= 1:
            r = np.cumsum(ev) / np.sum(ev)
            ind = np.where(r >= energy)
            dim = ind[0]
        else:
            dim = energy

        w = v[:, 0:dim]

    ##
    w = np.asarray(w, dtype=x.dtype)

    return (is_mean, x_mean, is_norm,  x_norm, w, ev, dim)


def pca_te(x, proj):
    # x: n*d
    # x_mean: 1*d
    # x_norm: 1*d
    # w: d*d2
    # output: n*d2
    is_mean, x_mean, is_norm, x_norm, w, _, _ = proj
    if is_mean:
        z = x - x_mean
    else:
        z = x

    if is_norm:
        z = z / x_norm

    return np.dot(z, w)


def prep_image(im, normal_height, normal_width, normal_type, is_swap_axis, interp='bilinear'):
    h, w, _ = im.shape
    if normal_type == 'keep_aspect_ratio':

        if h != normal_height or w != normal_width:
            r1 = 1. * normal_height / h
            r2 = 1. * normal_width / w
            if r1 > r2:
                nh = normal_height
                nw = np.floor(r1 * w + 0.5)
            elif r2 > r1:
                nw = normal_width
                nh = np.floor(r2 * h + 0.5)
            else:
                nh = normal_height
                nw = normal_width
            nh = np.int32(nh)
            nw = np.int32(nw)
            im = MISC.imresize(im, [nh, nw], interp=interp)

            # Central crop
            h, w, _ = im.shape
            im = im[h // 2 - normal_height // 2:h // 2 + normal_height //
                    2, w // 2 - normal_width // 2:w // 2 + normal_width // 2]

    elif normal_type == 'keep_all_content':
        if h != normal_height or w != normal_width:
            im = MISC.imresize(im, [normal_height, normal_width], interp='bilinear')
    else:
        print('normal_type error, please set <keep_aspect_ratio> or <keep_all_content>.')

    if is_swap_axis:
        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
        # Convert to BGR
        im = im[::-1, :, :]
    return im  # floatX(im[np.newaxis])


def create_dir(dir_name, folder=None):
    if folder == None:
        path = dir_name
    else:
        path = os.path.join(dir_name, folder)

    if not os.path.isdir(path):
        os.mkdir(path)

    return path


def get_path(dir_name, folder):

    return os.path.join(dir_name, folder)


def resize_batch(x, nm_h, nm_w, interp_type):
    # x: n*h*w*[c]
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    if h == nm_h and w == nm_w:
        return x
    if len(x.shape) == 4:
        y = np.zeros((n, nm_h, nm_w, x.shape[3]), dtype=np.float32)
        for ii in xrange(n):
            y[ii, :, :, :] = MISC.imresize(x[ii], [nm_h, nm_w], interp=interp_type)
    else:
        y = np.zeros((n, nm_h, nm_w), dtype=np.float32)
        for ii in xrange(n):
            y[ii, :, :] = MISC.imresize(x[ii], [nm_h, nm_w], interp=interp_type)
    return y


def get_patch_warping(img, ctr, tmp_sz, win_sz):

    img = np.float32(img)
    if len(img.shape) == 3:
        isColor = True
    else:
        isColor = False
    h = img.shape[0]
    w = img.shape[1]

    x = np.arange(1, win_sz[0] + 1) - win_sz[0] / 2 + 0.5
    y = np.arange(1, win_sz[1] + 1) - win_sz[1] / 2
    [x, y] = np.meshgrid(x, y)
    p3 = tmp_sz[0] / win_sz[0]
    # print p3,p3*tmp_sz[1]/win_sz[1]
    yp = ctr[1] + y * (p3 * tmp_sz[1] / win_sz[1]) - 1
    xp = ctr[0] + x * p3 - 1

    # save_mat_file('warping.mat',x,y,xp,yp) #??
    ##
    x0 = np.int32(xp)
    x1 = x0 + 1
    y0 = np.int32(yp)
    y1 = y0 + 1

    rx0 = xp - x0
    rx1 = 1 - rx0
    ry = yp - y0

    # --

    x0_bool = (x0 < 0) + (x0 > w - 1)
    x1_bool = (x1 < 0) + (x1 > w - 1)
    y0_bool = (y0 < 0) + (y0 > h - 1)
    y1_bool = (y1 < 0) + (y1 > h - 1)

    x0[x0_bool] = 0
    x1[x1_bool] = 0
    y0[y0_bool] = 0
    y1[y1_bool] = 0

    if isColor == True:
        patch = np.zeros((win_sz[1], win_sz[0], 3))
        for ii in range(3):
            patch[:, :, ii] = (rx1 * img[y0, x0, ii] * (~(y0_bool + x0_bool)) + rx0 * img[y0, x1, ii] * (~(y0_bool + x1_bool))) * (1 - ry) + \
                (rx1 * img[y1, x0, ii] * (~(y1_bool + x0_bool)) + rx0 * img[y1, x1, ii] * (~(y1_bool + x1_bool))) * ry
    else:
        patch = (rx1 * img[y0, x0] * (~(y0_bool + x0_bool)) + rx0 * img[y0, x1] * (~(y0_bool + x1_bool))) * (1 - ry) + \
            (rx1 * img[y1, x0] * (~(y1_bool + x0_bool)) + rx0 * img[y1, x1] * (~(y1_bool + x1_bool))) * ry

    patch[patch < 0] = 0
    patch[patch > 255] = 255
    return np.uint8(patch + 0.5)


def elewise_mlp_list(a, b):
    if len(a) != len(b) and len(a) != 1 and len(b) != 1:
        print('error: len(a)!=len(b).')
        ab = None
    elif len(a) == 1:
        ab = [a[0] * b[i] for i in range(len(b))]
    elif len(b) == 1:
        ab = [a[i] * b[0] for i in range(len(a))]
    else:
        ab = [a[i] * b[i] for i in range(len(a))]
    return ab


def elewise_div_list(a, b):
    if len(a) != len(b) and len(a) != 1 and len(b) != 1:
        print('error: len(a)!=len(b).')
        ab = None
    elif len(a) == 1:
        ab = [a[0] / b[i] for i in range(len(b))]
    elif len(b) == 1:
        ab = [a[i] / b[0] for i in range(len(a))]
    else:
        ab = [a[i] / b[i] for i in range(len(a))]
    return ab


def fun_rct2ctr(rcts):
    # rcts: [w,h,wwidth,hwidth]
    ctrs = np.copy(rcts)
    ctrs[:, 0] = rcts[:, 0] + rcts[:, 2] / 2
    ctrs[:, 1] = rcts[:, 1] + rcts[:, 3] / 2
    return ctrs


def fun_ctr2rct(ctrs):
    # ctrs: [cw,ch,wwidth,hwidth]
    rcts = np.copy(ctrs)
    rcts[:, 0] = ctrs[:, 0] - ctrs[:, 2] / 2
    rcts[:, 1] = ctrs[:, 1] - ctrs[:, 3] / 2
    return rcts


def read_image(fpath, isColor, isPad, padPos):
    '''
    isColor: need produce rgb channels
    isPad:   when isColor == True: gray image (2D) --> color image (3D)
    padPos: if isColor == True and isPad == True, for padPos ==0, we have output as c*h*w, for padPos == -1, h*w*c
    '''
    if isColor:
        img = np.array(Image.open(fpath))
        if len(img.shape) == 2 and isPad:
            assert(padPos == 0 or padPos == -1)
            if padPos == -1:
                img = np.dstack((img, img, img))
            if padPos == 0:
                img = np.expand_dims(img, axis=0)
                img = np.concatenate((img, img, img), axis=padPos)
        if len(img.shape) == 3:
            assert(padPos == 0 or padPos == -1)
            if padPos == 0:
                img = np.moveaxes(img, [0, 1, 2], [2, 0, 1])
    else:
        img = np.array(Image.open(fpath).convert('L'))
    return img


def get_layers_lr_wgt_mul(netparams):
    lr_mul = []
    wgt_mul = []
    for ii in xrange(len(netparams)):
        if netparams[ii].has_key('lr_mul'):
            lr_mul = np.append(lr_mul, netparams[ii]['lr_mul'])
            wgt_mul = np.append(wgt_mul, netparams[ii]['weight_mul'])
    lr_mul = np.float32(lr_mul)
    wgt_mul = np.float32(wgt_mul)
    return lr_mul, wgt_mul


def get_net_layerNo(netparams, keyname):
    flag = False
    for ii in xrange(len(netparams)):
        if netparams[ii]['key'] == keyname:
            flag = True
            break
    if flag:
        return ii
    else:
        print(keyname)
        return -2


# use after parsing
def net_params_print(netparams, logger):
    nlayer = len(netparams)
    str = '-- net: %d layers --' % nlayer
    logger.info(str)
    str = ''
    for ii in xrange(nlayer):
        keyname = netparams[ii]['key']
        if netparams[ii]['name'] == 'conv':
            kernel = netparams[ii]['kernel']
            conv_stride = netparams[ii]['conv_stride']
            conv_pad = netparams[ii]['conv_pad']

            str_kernel = vector2string(kernel, 'int')
            str_conv_stride = vector2string(conv_stride, 'int')
            str_conv_pad = vector2string(conv_pad, 'int')
            str = str + '\n\t**[%2d--%s] conv layer: \n\t\t<%s>\t<%s>\t<%s>' % (ii, keyname, str_kernel,
                                                                                str_conv_stride, str_conv_pad
                                                                                )
        elif netparams[ii]['name'] == 'convlinear':
            kernel = netparams[ii]['kernel']
            conv_stride = netparams[ii]['conv_stride']
            conv_pad = netparams[ii]['conv_pad']

            str_kernel = vector2string(kernel, 'int')
            str_conv_stride = vector2string(conv_stride, 'int')
            str_conv_pad = vector2string(conv_pad, 'int')
            str = str + '\n\t**[%2d--%s] convlinear layer: \n\t\t<%s>\t<%s>\t<%s>' % (ii, keyname, str_kernel,
                                                                                      str_conv_stride, str_conv_pad
                                                                                      )

        elif netparams[ii]['name'] == 'convpool':
            kernel = netparams[ii]['kernel']
            conv_stride = netparams[ii]['conv_stride']
            conv_pad = netparams[ii]['conv_pad']
            pool_size = netparams[ii]['pool_size']
            pool_pad = netparams[ii]['pool_pad']

            str_kernel = vector2string(kernel, 'int')
            str_conv_stride = vector2string(conv_stride, 'int')
            str_pool_size = vector2string(pool_size, 'int')
            str_conv_pad = vector2string(conv_pad, 'int')
            str_pool_pad = vector2string(pool_pad, 'int')
            str = str + '\n\t**[%2d--%s] convpool layer: \n\t\t<%s>\t<%s>\t<%s>\n\t\t<%s>\t<%s>' % (ii, keyname, str_kernel,
                                                                                                    str_conv_stride, str_conv_pad,
                                                                                                    str_pool_size, str_pool_pad)
        elif netparams[ii]['name'] == 'convdroppool':
            kernel = netparams[ii]['kernel']
            conv_stride = netparams[ii]['conv_stride']
            conv_pad = netparams[ii]['conv_pad']
            pool_size = netparams[ii]['pool_size']
            pool_pad = netparams[ii]['pool_pad']
            drop_rate = netparams[ii]['drop_rate']

            str_kernel = vector2string(kernel, 'int')
            str_conv_stride = vector2string(conv_stride, 'int')
            str_pool_size = vector2string(pool_size, 'int')
            str_conv_pad = vector2string(conv_pad, 'int')
            str_pool_pad = vector2string(pool_pad, 'int')
            str_drop_rate = '%.4f' % drop_rate
            # print str_drop_rate
            str = str + '\n\t**[%2d--%s] convdroppool layer: \n\t\t<%s>\t<%s>\t<%s>\n\t\t<%s>\t<%s>\n\t\t<%s>' % (ii, keyname, str_kernel,
                                                                                                                  str_conv_stride, str_conv_pad,
                                                                                                                  str_pool_size, str_pool_pad, str_drop_rate)
        elif netparams[ii]['name'] == 'input':
            str = str + '\n\t**%2d input layer:' % ii
        elif netparams[ii]['name'] == 'flat':
            str = str + '\n\t**%2d flat layer:' % ii
        elif netparams[ii]['name'] == 'swapdim':
            str = str + '\n\t**%2d swapdim layer:' % ii
        elif netparams[ii]['name'] == 'rnn':
            hid_dim = netparams[ii]['hid_dim']
            str = str + '\n\t**[%2d--%s] rnn   layer: \n\t\t<%4d> ' % (ii, keyname, hid_dim)
        elif netparams[ii]['name'] == 'strnn':
            hid_dim = netparams[ii]['hid_dim']
            layername = netparams[ii]['hid_dim']
            direction = netparams[ii]['direction']
            str = str + '\n\t**[%2d--%s] strnn   layer <%s>: \n\t\t<%s>\t<%4d> ' % (ii, keyname, layername, direction, hid_dim)
        elif netparams[ii]['name'] == 'srnn':
            hid_dim = netparams[ii]['hid_dim']
            layername = netparams[ii]['hid_dim']
            direction = netparams[ii]['direction']
            str = str + '\n\t**[%2d--%s] srnn   layer <%s>: \n\t\t<%s>\t<%4d> ' % (ii, keyname, layername, direction, hid_dim)
        elif netparams[ii]['name'] == 'dropout':
            drop_rate = netparams[ii]['drop_rate']
            str = str + '\n\t**[%2d--%s] rnn   layer: \n\t\t<%.4f> ' % (ii, keyname, drop_rate)
        elif netparams[ii]['name'] == 'align':
            hid_dim = netparams[ii]['hid_dim']
            str = str + '\n\t**[%2d--%s] align layer: \n\t\t<%4d> ' % (ii, keyname, hid_dim)
        elif netparams[ii]['name'] == 'channel':
            #class_num = netparams[ii]['class_num']
            str = str + '\n\t**[%2d--%s] channel layer: ' % (ii, keyname)
        elif netparams[ii]['name'] == 'fc':
            out_dim = netparams[ii]['out_dim']
            str = str + '\n\t**[%2d--%s] fc    layer: \n\t\t<%4d> ' % (ii, keyname, out_dim)
        elif netparams[ii]['name'] == 'logreg':
            class_num = netparams[ii]['class_num']
            str = str + '\n\t**[%2d--%s] loss  layer: \n\t\t<%4d> ' % (ii, keyname, class_num)
        elif netparams[ii]['name'] == 'softmax':
            str = str + '\n\t**[%2d--%s] softmax  layer: ' % (ii, keyname)
        elif netparams[ii]['name'] == 'meanshift':
            kernel = netparams[ii]['kernel']
            stride = (1, 1)

            str_kernel = vector2string(kernel, 'int')
            str_stride = vector2string(stride, 'int')
            str = str + '\n\t**%2d meanshift layer: \n\t\t<%s>\n\t\t<%s>' % (ii, str_kernel, str_stride)
        else:
            print('error')
        str_in_shape = vector2string(netparams[ii]['in_shape'], 'int')
        str_out_shape = vector2string(netparams[ii]['out_shape'], 'int')
        toplayer = netparams[ii]['top']
        str = str + '\n\t\t[ --%s ]\n\t\t--inshape:  %s \n\t\t--outshape: %s ' % (toplayer, str_in_shape, str_out_shape)
    logger.info(str)


# netparams: list of dics
# input_shape: (batch_size,channel,height,width)
def net_params_parsing(netparams):
    nlayer = len(netparams)
    for ii in xrange(nlayer):
        if ii != 0:  # ??
            kk = get_net_layerNo(netparams, netparams[ii]['top'])
            netparams[ii]['in_shape'] = netparams[kk]['out_shape']
        in_shape = netparams[ii]['in_shape']

        if netparams[ii]['name'] == 'conv' or netparams[ii]['name'] == 'convlinear':
            kernel = netparams[ii]['kernel']
            conv_stride = netparams[ii]['conv_stride']
            out_shape, conv_pad = get_conv_out_shape(in_shape, kernel, conv_stride)
            netparams[ii]['conv_pad'] = conv_pad
        elif netparams[ii]['name'] == 'convpool' or netparams[ii]['name'] == 'convdroppool':
            kernel = netparams[ii]['kernel']
            conv_stride = netparams[ii]['conv_stride']
            pool_size = netparams[ii]['pool_size']
            pool_stride = netparams[ii]['pool_stride']
            out_shape, conv_pad, pool_pad = get_convpool_out_shape(in_shape, kernel, conv_stride, pool_size, pool_stride)
            netparams[ii]['conv_pad'] = conv_pad
            netparams[ii]['pool_pad'] = pool_pad
        elif netparams[ii]['name'] == 'flat':
            out_shape = (in_shape[0], np.prod(in_shape[1:]))
        elif netparams[ii]['name'] == 'dropout':
            out_shape = in_shape
        elif netparams[ii]['name'] == 'rnn':
            hid_dim = netparams[ii]['hid_dim']
            out_shape = (in_shape[0], 1, in_shape[2], hid_dim)
        elif netparams[ii]['name'] == 'strnn' or netparams[ii]['name'] == 'srnn':
            hid_dim = netparams[ii]['hid_dim']
            out_shape = (in_shape[0], hid_dim, in_shape[2], in_shape[3])
        elif netparams[ii]['name'] == 'channel':
            out_shape = (in_shape[0] * in_shape[2] * in_shape[3], in_shape[1])
        elif netparams[ii]['name'] == 'align':
            out_shape = (in_shape[0], in_shape[3])
        elif netparams[ii]['name'] == 'swapdim':
            out_shape = (in_shape[0], in_shape[2], in_shape[1], in_shape[3])
        elif netparams[ii]['name'] == 'fc':
            out_dim = netparams[ii]['out_dim']
            out_shape = (in_shape[0], out_dim)
        elif netparams[ii]['name'] == 'logreg':
            class_num = netparams[ii]['class_num']
            out_shape = (in_shape[0], class_num)
        elif netparams[ii]['name'] == 'meanshift':
            kernel = netparams[ii]['kernel']
            stride = (1, 1)
            poolsize = (1, 1)
            out_shape = get_convpool_out_shape(in_shape, kernel, stride, poolsize)
        elif netparams[ii]['name'] == 'input':
            out_shape = in_shape
        elif netparams[ii]['name'] == 'softmax':
            out_shape = (in_shape[0], 1, in_shape[2], in_shape[3])
        else:
            print('error')
        netparams[ii]['out_shape'] = out_shape

        # if netparams[ii].has_key('layer') and netparams[ii-1].has_key('layer'):
        #    if netparams[ii]['layer'] == netparams[ii-1]['layer']:
        #	in_shape = netparams[ii-1]['in_shape']
        #        out_shape = netparams[ii-1]['out_shape']
    return netparams


def get_conv_out_shape(inshape, kernel, conv_stride):
    # print inshape,kernel,stride
    conv_pad = (kernel[2] // 2, kernel[3] // 2)
    h = (np.arange(0, inshape[2] - kernel[2] + 1 + 2 * conv_pad[0], conv_stride[0])).shape[0]
    w = (np.arange(0, inshape[3] - kernel[3] + 1 + 2 * conv_pad[1], conv_stride[1])).shape[0]
    out_shape = (inshape[0], kernel[0], h, w)
    return out_shape, conv_pad


def get_convpool_out_shape(inshape, kernel, conv_stride, pool_size, pool_stride):
    # print inshape,kernel,stride
    conv_pad = (kernel[2] // 2, kernel[3] // 2)
    h = (np.arange(0, inshape[2] - kernel[2] + 1 + 2 * conv_pad[0], conv_stride[0])).shape[0]
    w = (np.arange(0, inshape[3] - kernel[3] + 1 + 2 * conv_pad[1], conv_stride[1])).shape[0]
    pool_pad = (0, 0)  # (pool_size[0]//2,pool_size[1]//2)
    h = (np.arange(0, h - pool_size[0] + 1 + 2 * pool_pad[0], pool_stride[0])).shape[0]
    w = (np.arange(0, w - pool_size[1] + 1 + 2 * pool_pad[1], pool_stride[1])).shape[0]
    #h = h//poolsize[0]
    #w = w//poolsize[1]
    out_shape = (inshape[0], kernel[0], h, w)
    return out_shape, conv_pad, pool_pad


def matrix2string(x, datatype):
    h = x.shape[0]
    w = x.shape[1]
    str = '\n[\n'
    for ii in xrange(h):
        for jj in xrange(w):
            if datatype == 'int':
                s = '%6d, ' % (x[ii, jj])
            elif datatype == 'float':
                s = '%6.4f, ' % (x[ii, jj])
            str = str + s
        str = str + '\n'
    str = str + '\n]\n'
    return str


def vector2string(x, datatype):
    n = len(x)
    str = ''
    for ii in range(n):
        if datatype == 'int':
            s = '%6d, ' % (x[ii])
        elif datatype == 'float':
            s = '%6.4f, ' % (x[ii])
        str = str + s
    str = str + ''
    return str


def get_tuple_from_tuple(x, idx):
    y = []
    for ii in idx:
        y.append(x[ii])
    return y


def locations_of_substring(string, substring):
    """Return a list of locations of a substring."""

    substring_length = len(substring)

    def recurse(locations_found, start):
        location = string.find(substring, start)
        if location != -1:
            return recurse(locations_found + [location], location + substring_length)
        else:
            return locations_found

    return recurse([], 0)


def read_filenames(fpath, filetype):
    filepaths = glob.glob(fpath + '/*' + filetype)
    nfilepath = len(filepaths)
    filepaths = sorted(filepaths, key=str.lower)
    _, files = split_fpaths(filepaths)
    return filepaths, files, nfilepath


def split_fpaths(fpathfiles):
    n = len(fpathfiles)
    fpaths = []
    fnames = []
    for ii in xrange(n):
        fpath, fname = os.path.split(fpathfiles[ii])
        fpaths.append(fpath)
        fnames.append(fname)
    return fpaths, fnames


def save_mat_file(filename, data1, data2, data3, data4):
    mdic = {'data1': data1}
    if data2 is not None:
        mdic['data2'] = data2
    if data3 is not None:
        mdic['data3'] = data3
    if data4 is not None:
        mdic['data4'] = data4
    # print mdic
    scipy.io.savemat(filename, mdic)


def save_variable_list(x, filepath, is_shared):
    if is_shared == 1:
        y = []
        for xi in x:
            y.append(xi.get_value())
        fp = open(filepath, "wb")
        cPickle.dump(y, fp, protocol=-1)
        fp.close()
    else:
        fp = open(filepath, "wb")
        cPickle.dump(x, fp, protocol=-1)
        fp.close()


def load_variable_list(filepath):
    fp = open(filepath, "rb")
    y = cPickle.load(fp)
    fp.close()
    return y


def infer_pred_shape(gt_lmks, ref_point, ref_width):
    n = gt_lmks.shape[0]
    n_lmk = gt_lmks.shape[1] / 2
    idx_h = np.arange(0, n_lmk * 2, 2)
    idx_w = np.arange(1, n_lmk * 2, 2)
    rand_idx = np.random.permutation(n)
    pred_lmks = np.zeros((n, n_lmk * 2), dtype=np.float32)
    pred_lmks[:, idx_h] = (gt_lmks[:, idx_h] - ref_point[:, 0].reshape(-1, 1)) / ref_width
    pred_lmks[:, idx_w] = (gt_lmks[:, idx_w] - ref_point[:, 1].reshape(-1, 1)) / ref_width
    pred_lmks = pred_lmks[rand_idx, :];
    pred_lmks[:, idx_h] = pred_lmks[:, idx_h] * ref_width + ref_point[:, 0].reshape(-1, 1)
    pred_lmks[:, idx_w] = pred_lmks[:, idx_w] * ref_width + ref_point[:, 1].reshape(-1, 1)

    return pred_lmks


def kmeans_cluster(x, k, isDataNorm, isCtrNorm):
    # x: n*d
    if isDataNorm:
        nm = np.sqrt(np.sum(x**2, axis=-1, keepdims=True))
        x = x / nm
    # no whiten
    centriod, label = cluster.vq.kmeans2(x, k, iter=500, minit='points')
    if isCtrNorm:
        nm = np.sqrt(np.sum(centriod**2, axis=-1, keepdims=True))
        centriod = centriod / nm
    return centriod, label

##


def swap_columns(x, i1, i2):
    temp = np.copy(x[:, i1])
    x[:, i1] = x[:, i2]
    x[:, i2] = temp
    return x


def save_vars6_dumps(filepath, x1, x2, x3, x4, x5, x6):
    fp = open(filepath, "wb")
    cPickle.dump(x1, fp, protocol=-1)
    if x2 is not None:
        cPickle.dump(x2, fp, protocol=-1)
    if x3 is not None:
        cPickle.dump(x3, fp, protocol=-1)
    if x4 is not None:
        cPickle.dump(x4, fp, protocol=-1)
    if x5 is not None:
        cPickle.dump(x5, fp, protocol=-1)
    if x6 is not None:
        cPickle.dump(x6, fp, protocol=-1)
    fp.close()


def load_vars6_dumps(filepath, n):
    fp = open(filepath, "rb")
    x1 = cPickle.load(fp)
    if n == 1:
        fp.close()
        return x1

    x2 = cPickle.load(fp)
    if n == 2:
        fp.close()
        return x1, x2

    x3 = cPickle.load(fp)
    if n == 3:
        fp.close()
        return x1, x2, x3

    x4 = cPickle.load(fp)
    if n == 4:
        fp.close()
        return x1, x2, x3, x4

    x5 = cPickle.load(fp)
    if n == 5:
        fp.close()
        return x1, x2, x3, x4, x5

    x6 = cPickle.load(fp)
    if n == 6:
        fp.close()
        return x1, x2, x3, x4, x5, x6


def build_w_b_kernel(rng, kernel, str_type, factor=1.):
    fan_in = np.prod(kernel[1:])
    fan_out = kernel[0] * np.prod(kernel[2:])

    if str_type == 'uniform':
        thre = np.sqrt(6. / (fan_in + fan_out)) * factor
        W = np.asarray(rng.uniform(size=kernel, low=-thre, high=thre), dtype=theano.config.floatX)
        b = np.zeros((kernel[0],), dtype=theano.config.floatX)
    elif str_type == 'gaussian':
        W = np.asarray(rng.normal(loc=factor[0], scale=factor[1], size=kernel), dtype=theano.config.floatX)
        b = np.zeros((kernel[0],), dtype=theano.config.floatX)
    return W, b


'''
  For a :class:`DenseLayer <lasagne.layers.DenseLayer>`, if ``gain='relu'``
    and ``initializer=Uniform``, the weights are initialized as
    .. math::
       a &= \\sqrt{\\frac{12}{fan_{in}+fan_{out}}}\\\\
       W &\sim U[-a, a]
    If ``gain=1`` and ``initializer=Normal``, the weights are initialized as
    .. math::
       \\sigma &= \\sqrt{\\frac{2}{fan_{in}+fan_{out}}}\\\\
       W &\sim N(0, \\sigma)
'''


def build_w_b(rng, n_in, n_out, str_type, factor):
    if str_type == 'uniform':
        thre = np.sqrt(12. / (n_in + n_out)) * factor
        # print thre
        W = np.asarray(rng.uniform(size=(n_in, n_out),
                                   low=-thre, high=thre), dtype=theano.config.floatX)
        b = np.zeros((n_out,), dtype=theano.config.floatX)
        #b = b
    elif str_type == 'zeros':
        W = np.asarray(np.zeros((n_in, n_out)), dtype=theano.config.floatX)
        b = np.zeros((n_out,), dtype=theano.config.floatX)
    elif str_type == 'identity':
        thre = factor
        W = np.asarray(np.eye(n_in) * thre, dtype=theano.config.floatX)
        b = np.zeros((n_out,), dtype=theano.config.floatX)
    elif str_type == 'gaussian':
        W = np.asarray(rng.normal(loc=factor[0], scale=factor[1], size=(n_in, n_out)), dtype=theano.config.floatX)
        b = np.zeros((n_out,), dtype=theano.config.floatX)
    return W, b
