#encoding:utf-8

import argparse
import tf_utis
from tf_utis import *
from configer import *

parser = argparse.ArgumentParser()
parser.add_argument('--start','-s',default=0,type=int)
parser.add_argument('--end','-e',default=100,type=int)
parser.add_argument('--gpu','-g',default=0,type=str)
parser.add_argument('--seq','-seq',default=None,type=str)
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
####################################################
#####               Main Fun            ############
####################################################
if __name__ == '__main__':
    ####
    if not os.path.isdir(data_path):
        raise Exception('data not exist',data_path)

    #### 
    fdrs = next(os.walk(data_path))[1]
    fdrs = sorted(fdrs)
    n_fdr = len(fdrs)
    #### build VGG model
    with tf.device(gpu_id):
        vgg = vgg19_tf.Vgg19( vgg_model_path, vgg_out_layers)
        vgg_sess = tf.Session(config = config)
        pca_te = tf_utis.PCA()
        pca_sess = tf.Session(config = config)
        res_map = tf_utis.Response()
        res_sess = tf.Session(config = config)
    vgg_map_total, vgg_map_idx, vgg_map_nlayer = vgg.out_map_total, vgg.out_map_idx, vgg.out_map_nlayer
    
    ####
    for ifdr in range(opt.start,opt.end):#np.arange (start_sample,end_sample,step_sample):#n_fdr
	##
        fdr = fdrs[ifdr]
        if opt.seq is not None:
            if fdr != opt.seq:
                continue
	#    continue
        fpath = os.path.join(data_path,fdr)
        f_rcts = glob.glob(fpath+'/groundtruth_rect*.txt')
        f_imgs = glob.glob(fpath+'/img/*.jpg')
        n_img = len(f_imgs)
        n_rct = len(f_rcts)
        f_imgs = sorted(f_imgs,key=str.lower)

        print("{}:{}:{}".format(ifdr, fdr, n_img))
	## read images >> X0
        # n_img = 30
        for ii in range(n_img):
            img = read_image(f_imgs[ii],True,True,-1)
            if ii == 0:
                im_sz = np.asarray([img.shape[1],img.shape[0]])
                X0 = np.zeros((n_img,img.shape[0],img.shape[1],3),dtype=np.uint8)
            X0[ii,:,:,:] = img
        del img		

	################# each sequence ##############################
        for iseq in range(n_rct):
            str1 = 'result_%s_%d_%.3f.mat' %(fdr,iseq,update_factor)
            fname = os.path.join(cache_path,str1)
            if os.path.isfile(fname):
                print("{} existed result".format(fname))
                continue
	
	    #### log file
            str1 = 'log_%s_%s_%d.txt' %(pstr,fdr,iseq) #pstr=gcnn
            log_file = os.path.join(cache_path,str1)
            logger = Logger(logname=log_file, loglevel=1, logger=">>").getlog()
	    
	    #### load rct and convert it ctr style 
            gt_rcts0 = np.loadtxt(f_rcts[iseq],delimiter=',')
            gt_rcts = np.floor(fun_rct2ctr(gt_rcts0)) 
 
	    #### set peak map
            target_sz = gt_rcts[0,2:]
            window_sz,padding_h, padding_w = fun_get_search_window2(target_sz,im_sz,None,None) 

            cell_size=np.prod(window_sz)/(fea_sz[0]*fea_sz[1])
            if cell_size == 0:
                cell_size = 1
            print("cell_size:{}".format(cell_size))

            pmap,pmap_ctr_h,pmap_ctr_w = fun_get_peak_map(window_sz,target_sz,cell_size,fea_sz,False)
            assert(pmap.shape[0]==fea_sz[0] and pmap.shape[1]==fea_sz[1])
            str1 = "target_sz: [%d, %d], window_sz: [%d, %d], pmap.shape: [%d, %d], cellsize: [%d] " %(target_sz[0], target_sz[1],\
                     window_sz[0], window_sz[1], pmap.shape[0], pmap.shape[1],  cell_size)
            logger.info(str1)

            prod_hw = fea_sz[1]*fea_sz[0]
            y = np.expand_dims(pmap,axis=0)
            y = np.expand_dims(y,axis=0)
            yf = np.fft.fft2(y,axes=(-2,-1))

	    #### cos_win
            cos_win = fun_cos_win(fea_sz[1],fea_sz[0],(1,fea_sz[1],fea_sz[0],1))*1.0e-3#/vgg_map_total

	    ####
            in_shape = (1,fea_sz[1],fea_sz[0],vgg_map_total)
            vgg_fea    = np.zeros(in_shape,dtype=np.float32) 
            ####
            model_alphaf = np.zeros((nn_p,fea_sz[1],fea_sz[0]),dtype=np.complex128)
            model_xf = np.zeros((nn_p,1,100,fea_sz[1],fea_sz[0]),dtype=np.complex128)
	    ####
            pred_rcts  = np.zeros((n_img,4),dtype=np.float32)
            pred_rcts[0,:] = np.copy(gt_rcts[0,:])
            cur_rct = np.copy(pred_rcts[0,:])
	    
            save_fdr = '%s_%d' %(fdr,iseq)
            save_path = os.path.join(cache_path,save_fdr)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

	    ############################## extract feature ##########################
            def extract_feature(ims,ctr_rcts,cos_win,padding_hw):
                padding_h = padding_hw[0]
                padding_w = padding_hw[1]
                n = ctr_rcts.shape[0]
                l = len(ims.shape)
                # pdb.set_trace()
                if l == 4:
                    assert(ims.shape[0]==n)
                ## crop out patch
                patches = []
                for ii in range(n):
                    # window_sz,_,_ = fun_get_search_window2(ctr_rcts[ii,2:],None,padding_h,padding_w) # ??
                    window_sz,_,_ = fun_get_search_window2(ctr_rcts[ii,2:],None,padding_h,padding_w)
                    if l==4: 
                        patch = fun_get_patch(np.copy(ims[ii]),ctr_rcts[ii,0:2],window_sz)
                    else:
                        # pdb.set_trace()
                        patch = fun_get_patch(np.copy(ims),ctr_rcts[ii,0:2],window_sz)
                    patches.append(patch)
                patches = vgg_process_images(patches,**img_param)     

                feed_dict = {vgg.images: patches, vgg.nscale: 1,vgg.cos_win:cos_win}
                if n != 1:
                    feed_dict1 = {vgg.images: patches, vgg.nscale: 7,vgg.cos_win:cos_win}
                    vgg_fea1 = vgg_sess.run(vgg.vgg_fea1, feed_dict=feed_dict1)
                else:
                    vgg_fea1 = vgg_sess.run(vgg.vgg_fea1, feed_dict=feed_dict)
                return vgg_fea1
            
            #########################
            nn_d = nn_p*pca_energy #vgg_map_total
            nn_m = prod_hw
            assert(nn_d%nn_p ==0)
            nn_map = np.int32(nn_d/nn_p)
            
	    ####
            flag_occ = 0
            for jj in range(n_img):
		#### sampling patch
                im = np.copy(X0[jj,:,:,:])

		#################################################################
		##################### predict process ###########################
		#################################################################
                if jj > 0:
                    # wsz,_,_ = fun_get_search_window2(cur_rct[2:],None,padding_h,padding_w)
                    wsz,padding_h,padding_w = fun_get_search_window2(cur_rct[2:],im_sz,None,None)
                    padding_hw = np.array([padding_h,padding_w],dtype = np.float32)
                    search_offset = fun_get_search_ctr(wsz,factor=0.4)
                    noffset      = search_offset.shape[0]
                    nscale = search_scale.shape[0]
                    tmp_rcts = np.zeros((noffset*nscale,4))
                    tmp_pred_rcts = np.zeros((n_img,nn_p,4),dtype = np.float32)
                    response = np.zeros((noffset,nn_p,nscale,fea_sz[1],fea_sz[0]))
                    
                    count = 0
                    for ioffset in range(noffset):
                        ctr0 = np.floor(search_offset[ioffset,0:2] + cur_rct[0:2]+0.5)
                        for iscale in range(nscale):
                            tmp_rcts[count,0:2] = np.copy(ctr0)
                            tmp_rcts[count,2:] = cur_rct[2:]*search_scale[iscale,:]
                            count = count + 1
                    if jj == 1:
                        test_vgg_fea   = np.zeros((noffset*nscale,in_shape[1],in_shape[2],in_shape[3]),dtype=np.float32)
                        print("{}:{}:{}".format( search_scale, nscale, search_offset, noffset))
                    test_vgg_fea = extract_feature(np.copy(im),tmp_rcts,cos_win,padding_hw)  
                    
                    
                    feed_dict = {pca_te.is_mean:pca_is_mean,pca_te.is_norm:pca_is_norm,pca_te.x_mean:pca_x_mean,\
                                pca_te.x_norm:pca_x_norm,pca_te.w:pca_w,pca_te.vgg_fea:test_vgg_fea}
                    vgg_fea_pca = pca_sess.run(pca_te.vgg_fea_pca,feed_dict=feed_dict)

                    feed_dict = {res_map.vgg_fea_pca:vgg_fea_pca,res_map.model_alphaf:model_alphaf,res_map.model_xf:model_xf}
                    response = res_sess.run(res_map.response,feed_dict=feed_dict)
                    mx_offset = 0 #get_max_offset(response)
                    mx_scale,maxres = get_max_scale(response[mx_offset])
                    mx_layer = 0
                    mxres0 = np.zeros(nn_p)
                    mx_hh = np.zeros(nn_p)
                    mx_ww = np.zeros(nn_p)
                    for ilayer in range(nn_p):
                        mxres0[ilayer],mx_hh[ilayer],mx_ww[ilayer] = get_max_ps(response[mx_offset,mx_scale[ilayer],ilayer,:,:],pmap_ctr_h,pmap_ctr_w)
                        tmp_pred_rcts[jj,ilayer,2:] = cur_rct[2:]*search_scale[mx_scale[ilayer],:]
                        window_sz,_,_ = fun_get_search_window2(tmp_pred_rcts[jj,ilayer,2:],None,padding_h,padding_w)
                        ratio = 1.0 * window_sz/wsz
                        tmp_pred_rcts[jj,ilayer,2:] = cur_rct[2:]*ratio
                        tmp_pred_rcts[jj,ilayer,0:2] = cur_rct[0:2] + 1.0*np.asarray([mx_ww[ilayer],mx_hh[ilayer]])*window_sz/fea_sz + search_offset[mx_offset,:]
                        tmp_pred_rcts[jj,ilayer,:] = fun_border_modification(np.copy(tmp_pred_rcts[jj,ilayer,:]),im.shape[0],im.shape[1])
                    pred_rcts[jj,:] = np.mean(tmp_pred_rcts[jj,:,:],axis = 0)
                    cur_rct = np.copy(pred_rcts[jj,:])

                    ####
                    str1 = "[%3d-%3d/%3d]:[%3.2f,%3.2f,%3.2f,%3.2f],[%3.2f,%3.2f,%3.2f,%3.2f],[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f][%.2f,%.2f,%.2f], [%d, %.3f]\n\t\t[%d,%.4f,%.2f,%.2f]\n\t\t%s\n\t\t%s\n\t\t%s" %(jj,n_img,ifdr,\
                                      gt_rcts[jj,0],gt_rcts[jj,1],gt_rcts[jj,2],gt_rcts[jj,3],\
                                      pred_rcts[jj,0], pred_rcts[jj,1],pred_rcts[jj,2],pred_rcts[jj,3],\
                                      search_scale[mx_scale[0],0],search_scale[mx_scale[1],0],search_scale[mx_scale[2],0],\
                                      search_scale[mx_scale[3],0],search_scale[mx_scale[4],0],search_scale[mx_scale[5],0],\
                                      mx_offset,search_scale[mx_scale[0],0],search_scale[mx_scale[0],1],flag_occ,update_factor,\
                                      mx_layer,mxres0[0], mx_ww[0], mx_hh[0], \
                                      vector2string(mx_ww,'float'),\
                                      vector2string(mx_hh,'float'),\
                                      vector2string(mxres0,'float'))

                    logger.info(str1)
                    flag_occ = 0

                    str1 = '%04d.jpg' %(jj) #'T_%d.jpg'
                    fname = os.path.join(save_path,str1)
                    fun_draw_rct_on_image(X0[jj,:,:],fname,gt_rcts[jj,:],None,pred_rcts[jj,:])

                    str1 = 'T_%d_mask.jpg' %(jj)
                    fname = os.path.join(save_path,str1)

                    str1 = 'prop_tr_%s_%d_%d.mat' %(fdr,iseq,jj)
                    fname = os.path.join(save_path,str1)

 
		#################################################################
		########################### Preparing ###########################
		#################################################################
                window_sz,padding_h,padding_w = fun_get_search_window2(cur_rct[2:],im_sz,None,None)
                padding_hw = np.array([padding_h,padding_w],dtype=np.float32)
                vgg_fea = extract_feature(np.copy(im),np.expand_dims(cur_rct,0),cos_win,padding_hw)
                if jj == 0: # pca
                    pca_projections = fea_pca_tr(np.copy(vgg_fea[0]), nn_p, pca_energy, pca_is_mean, pca_is_norm)
                    pca_x_mean = np.zeros((nn_p,1,512),dtype = np.float32)
                    pca_x_norm = np.zeros((nn_p),dtype = np.float32)
                    pca_w = np.zeros((nn_p,512,pca_energy),dtype = np.float32)
                    for itracker in range(nn_p):
                        pca_x_mean[itracker,:,:] = pca_projections[itracker][1]
                        pca_x_norm[itracker] = pca_projections[itracker][3]
                        pca_w[itracker,:,:] = pca_projections[itracker][4]
                vgg_fea2 = fea_pca_te(np.copy(vgg_fea[0]), nn_p, pca_projections)

		################### update model  ###################################
                if (jj%cf_nframe_update == 0 or jj < 5): 
                      for kk in range(nn_p):
                        vgg_fea2[kk]=np.reshape(vgg_fea2[kk],(fea_sz[1],fea_sz[0],-1))
                        vgg_fea2[kk]=(np.expand_dims(vgg_fea2[kk],axis=0)).transpose(0,3,1,2)
                        xf = np.fft.fft2(vgg_fea2[kk],axes=(-2,-1))  
                        alphaf = fun_w(xf,yf,kernel_type,kernel_sigma,kernel_gamma) # h*w
                        if jj==0:
                            model_alphaf[kk] = np.copy(alphaf)
                            model_xf[kk,:,:,:,:] =np.copy(xf)
                        else:
                                model_alphaf[kk] = (1-update_factor)*model_alphaf[kk] + update_factor*alphaf
                                model_xf[kk]     = (1-update_factor)*model_xf[kk]    + update_factor*xf

 
	    ## save all results
            #sess_tr.close()
            #sess_te.close()
            if jj==n_img-1:
                pcs_loc_mean,pcs_loc_curv, pcs_loc_diff = fun_precision_location(pred_rcts,gt_rcts)
                pcs_olp_mean,pcs_olp_curv, pcs_olp_diff = fun_precision_overlap(pred_rcts,gt_rcts)
                str1 = '[%s, %d, %.3f]--[%.4f,%.4f]\n' %(fdr,iseq, update_factor, pcs_loc_mean,pcs_olp_mean),\
                                        np.array2string(pcs_loc_curv.transpose(),precision=4,separator=', '),\
                                        np.array2string(pcs_olp_curv.transpose(),precision=4,separator=', ') 
                logger.info(str1)
                close_logger(logger)

                str1 = 'result_%s_%d_%.3f.mat' %(fdr,iseq,update_factor)
                fname = os.path.join(cache_path,str1)
                save_mat_file(fname,gt_rcts,pred_rcts,pcs_loc_diff,pcs_olp_diff)
 
    vgg_sess.close()
    pca_sess.close()
    res_sess.close()
