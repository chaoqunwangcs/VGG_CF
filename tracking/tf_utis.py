import os
import tensorflow as tf
import numpy as np
from configer import *

# fea_sz = [57,57]
# kernel_sigma = 0.5
# kernel_type = 'linear'
# kernel_gamma = 1.0  # 1.e-6
# pca_energy = 100
# nn_p = 6  #
nscale = search_scale.shape[0]
M = pca_energy*fea_sz[0]*fea_sz[1]

class Response:
	def __init__(self):
		self.vgg_fea_pca = tf.placeholder(tf.float64,[7,nn_p,fea_sz[0]*fea_sz[1],pca_energy])	#7*6*3249*100
		self.model_alphaf = tf.placeholder(tf.complex64,[nn_p,fea_sz[0],fea_sz[1]])	#6*57*57
		self.model_xf = tf.placeholder(tf.complex64,[nn_p,1,pca_energy,fea_sz[0],fea_sz[1]])	#6*1*100*57*57

		with tf.name_scope("get_response"):
			self.build(self.vgg_fea_pca,self.model_alphaf,self.model_xf)

	def build(self,vgg_fea_pca,model_alphaf,model_xf):

		vgg_fea_pca = tf.transpose(vgg_fea_pca,[2,0,1,3])	#3249*7*6*100
		vgg_fea_pca = tf.reshape(vgg_fea_pca,[fea_sz[0],fea_sz[1],7,nn_p,pca_energy])	#57*57*7*6*100
		vgg_fea_pca = tf.transpose(vgg_fea_pca,perm=[2,3,4,0,1])	#7*6*100*57*57
		vgg_fea_pca = tf.cast(vgg_fea_pca,dtype=tf.complex64)
		model_xf = tf.transpose(model_xf,perm=[1,0,2,3,4])	#1*6*100*57*57

		zf = tf.fft2d(vgg_fea_pca)	#7*6*100*57*57
		k_zf_xf = tf.reduce_sum(tf.multiply(zf,tf.conj(model_xf)),axis=2)/M #7*6*57*57

		response = tf.real(tf.ifft2d(k_zf_xf * model_alphaf))
		self.response = tf.expand_dims(response,axis=0)


class PCA:
	def __init__(self):
		self.is_mean = tf.placeholder(tf.bool)
		self.x_mean = tf.placeholder(tf.float32,[nn_p,1,512])	#6*1*512
		self.is_norm = tf.placeholder(tf.bool)
		self.x_norm = tf.placeholder(tf.float32,[nn_p])	#6
		self.w = tf.placeholder(tf.float32,[nn_p,512,pca_energy])	#6*512*100

		self.vgg_fea = tf.placeholder(tf.float32,[nscale,fea_sz[0],fea_sz[1],512*nn_p])	#7*57*57*3072


		with tf.name_scope("Pca_te"):
			self.build(self.is_mean, self.x_mean, self.is_norm, self.x_norm, self.w, self.vgg_fea)

	def build(self, is_mean, x_mean, is_norm, x_norm, w, vgg_fea):

		x_mean = tf.tile(tf.expand_dims(x_mean,axis=0),multiples=[nscale,1,1,1])	#7*6*1*512
		self.tmp3 = x_mean
		x_norm = tf.expand_dims(x_norm,axis=0)
		x_norm = tf.expand_dims(x_norm,axis=0)
		x_norm = tf.expand_dims(x_norm,axis=0)
		x_norm = tf.transpose(x_norm,perm=[0,3,1,2])	#1*6*1*1

		w = tf.tile(tf.expand_dims(w,axis=0),multiples=[nscale,1,1,1])	#7*6*512*100
		vgg_fea = tf.transpose(vgg_fea,perm=[0,3,1,2])	#7*3072*57*57
		vgg_fea = tf.reshape(vgg_fea,[nscale,512*nn_p,fea_sz[0]*fea_sz[1]])	#7*3072*3249
		vgg_fea = tf.transpose(vgg_fea,perm=[1,0,2])	#3072*7*3249
		vgg_fea = tf.transpose(tf.reshape(vgg_fea,[nn_p,512,nscale,fea_sz[0]*fea_sz[1]]),perm=[2,0,3,1])	#7*6*3249*512

		z = tf.cond(is_mean, lambda: vgg_fea - x_mean, lambda: vgg_fea)
		z = tf.cond(is_norm, lambda: tf.divide(z,x_norm), lambda: z)
		self.vgg_fea_pca = tf.matmul(z,w)	#7*6*3249*100
