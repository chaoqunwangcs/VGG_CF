import skimage
import skimage.io
import skimage.transform
import numpy as np
import scipy.misc as MISC
from scipy import interpolate
import pdb  

# synset = [l.strip() for l in open('synset.txt').readlines()]

def vgg_resize_maps(map_list,normal_hw, interp = 'bilinear', outx = None):
    # map_list: list of 4d tesors [n,h,w,c_i] or 4d tensor [n,h,w,c]
    # maps: 4d tensor, n*nh*nw*\sum{c_i}\
    # pdb.set_trace()
    nmh,nmw = normal_hw
    assert(type(map_list) is list)
    #nlayer = len(map_list)
    n = map_list[0].shape[0]
    total_map = 0
    for imap in map_list:
        total_map += imap.shape[-1]

    if outx is None:
        outx = np.zeros((n,nmh,nmw,total_map),dtype = np.float32)
    count = 0
    if interp == 'bilinear':
        ip_type = 'linear'
    elif interp == 'bicubic':
        ip_type = 'cubic'
    else:
        assert(3==5)

    for imap in map_list:
        n,h,w,c = imap.shape
        
        h0_seq = np.arange(0,nmh-1.0e-6,nmh*1./h)
        w0_seq = np.arange(0,nmw-1.0e-6,nmw*1./w)
        h1_seq = np.arange(nmh)
        w1_seq = np.arange(nmw)

        for jj in range(c):
            for ii in range(n):
                #maps[ii,:,:,count] = vgg_process_one_image(imap[ii,:,:,jj],normal_height,normal_width,normal_type, False, interp_tool, interp)
                f = interpolate.interp2d(w0_seq,h0_seq,imap[ii,:,:,jj],kind=ip_type)
                outx[ii,:,:,count] = f(w1_seq,h1_seq)
            count += 1

    #return maps


def vgg_resize_image(im, nh, nw, interp_tool ='misc', interp = 'bilinear'):

    if interp_tool =='misc':
        im = MISC.imresize(im,[nh,nw],interp=interp)
    elif interp_tool == 'skimage':
        im = im / 255.0
        assert (0 <= im).all() and (im <= 1.0).all()
        im = skimage.transform.resize(im, (nh, nw))
        im = im*255.0
    else:
        assert(interp_tool=='misc' and interp_tool=='skimage')

    return im



def vgg_process_one_image(im,normal_height,normal_width,normal_type,is_swap_axis, interp_tool ='misc', interp = 'bilinear'):

    h = im.shape[0]
    w = im.shape[1]
    if normal_type == 'keep_aspect_ratio':

        if h!=normal_height or w!=normal_width:
            r1 = 1.*normal_height/h
            r2 = 1.*normal_width/w
            if r1 > r2:
                nh = normal_height
                nw = np.floor(r1*w + 0.5)
            elif r2 > r1:
                nw = normal_width
                nh = np.floor(r2*h + 0.5)
            else:
                nh = normal_height
                nw = normal_width
            nh = np.int32(nh)
            nw = np.int32(nw)

            im = vgg_resize_image(im, nh, nw, interp_tool, interp)

            # Central crop
            h, w, _ = im.shape
            im = im[h//2-normal_height//2:h//2+normal_height//2, w//2-normal_width//2:w//2+normal_width//2]

    elif normal_type == 'keep_all_content':
        if h!=normal_height or w!=normal_width:
            im = vgg_resize_image(im, normal_height,normal_width, interp_tool, interp)
            #MISC.imresize(im,[normal_height,normal_width],interp='bilinear')
    else:
        print('normal_type error, please set <keep_aspect_ratio> or <keep_all_content>.')

    if is_swap_axis:
        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
        # Convert to BGR
        im = im[::-1, :, :]
    return im #floatX(im[np.newaxis])

def vgg_process_images(ims,normal_hw,normal_type='keep_aspect_ratio',interp_tool ='misc', interp = 'bilinear'):
    # ims: list of 3d tesors [h,w,c] or 4d tensor [n,h,w,c]
    # ims_out: 4d tensor, n*nh*nw*nc
    normal_height,normal_width = normal_hw
    flag = type(ims) is list
    if flag:
        n = len(ims)
        c = ims[0].shape[-1]
    else:
        n, _, _, c = ims.shape
    ims_out = np.zeros((n,normal_height,normal_width,c),dtype = np.float32)
    for ii in range(n):
        h,w,_ = ims[ii].shape
        ims_out[ii] = vgg_process_one_image(ims[ii],normal_height,normal_width,normal_type, False, interp_tool, interp)

    return ims_out


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
