#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:10:32 2018

@author: lee
with batch normalization
"""
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import tifffile as tiff
from tensorflow.python.ops import gen_image_ops
import argparse
#import makedata3D as Input
#import makedata3D_test as InputT
import random

## directory setting
data_dir= 'I:\\Example\\Deep Learning\\'

## training, validating or testing
mode='training' ## or set as 'validating' with ground_truth, 'testing' without ground_truth

## parameter for train
train_iter_num=12000
train_batch_size=1
pre_train_batch_size=1

## parameter for validation
val_iter_num=120
val_batch_size=1

## parameters for testing
test_iter_num=120
test_batch_size=1

## set input image size
input_data_depth=280
input_data_width=204
input_data_height=328
input_data_channel=1


##### GPU SET
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

output_data_depth=input_data_depth
output_data_width=input_data_width
output_data_height=input_data_height
output_data_channel=1
EPS = 10e-5

input1_labels=[]
input2_labels=[]
gt_labels=[]
batch_index=0

def get_filenames(mode): #Get file name
    global input1_labels
    global input2_labels
    global gt_labels
    if (mode=='training' or mode=='validating'):
        input1_dir=data_dir+'input1/'
        input2_dir=data_dir+'input2/'
        gt_dir=data_dir+'ground truth/'
        input1_labels=os.listdir(input1_dir)
        input2_labels=os.listdir(input2_dir)
        gt_labels=os.listdir(gt_dir)
    elif mode=='testing':
        input1_dir=data_dir+'test/input1/'
        input2_dir=data_dir+'test/input2/'
        input1_labels=os.listdir(input1_dir)
        input2_labels=os.listdir(input2_dir)
    else: print('wrong mode!')


def get_data_tiff(mode, batch_size):
    global input1_labels
    global input2_labels
    global gt_labels
    global batch_index
    if len(input1_labels) == 0: get_filenames(mode)  
    maxlen = len(input1_labels)                          
#    print(maxlen)

    begin = batch_index                      
    end = batch_index + batch_size
    
    
    x1_data = np.array([], np.float32)
    x2_data = np.array([], np.float32)
    y_data = np.array([], np.float32)
    label_out=[]

    if (mode=='training' or mode=='validating'):
        for i in range(begin, end):
            Input1_Path = data_dir + 'input1/'+ input1_labels[i]
            Input2_Path = data_dir + 'input2/'+ input2_labels[i]
            GroundTruth_Path = data_dir + 'ground truth/'+gt_labels[i]
            label_out.append(input1_labels[i])
            input1_tif=tiff.imread(Input1_Path) 
            input2_tif=tiff.imread(Input2_Path) 
            input_GT=tiff.imread(GroundTruth_Path)
            inmin1, inmax1 = input1_tif.min(), input1_tif.max()
            normal_input1_tif=(input1_tif-inmin1)/(inmax1-inmin1)        
            inmin2, inmax2 = input2_tif.min(), input2_tif.max()
            normal_input2_tif=(input2_tif-inmin2)/(inmax2-inmin2)
            gtmin, gtmax = input_GT.min(), input_GT.max()
            normal_gt_tif=(input_GT-gtmin)/(gtmax-gtmin)
            ##x_data = np.append(x_data, output)  
            x1_data = np.append(x1_data, normal_input1_tif) 
            x2_data = np.append(x2_data, normal_input2_tif) 
            y_data = np.append(y_data, normal_gt_tif) 
        if end>=maxlen:
            end=maxlen
            batch_index = 0
        else:
            batch_index += batch_size  # update index for the next batch
        x1_data_ = x1_data.reshape(batch_size, input_data_depth * input_data_height * input_data_width )
        x2_data_ = x2_data.reshape(batch_size, input_data_depth * input_data_height * input_data_width )
        y_data_ = y_data.reshape(batch_size, input_data_depth * input_data_height * input_data_width )
        return x1_data_,x2_data_, y_data_ , label_out
    
    elif mode=='testing':
        for i in range(begin, end):
            Input1_Path = data_dir + 'test/input1/'+ input1_labels[i]
            Input2_Path = data_dir + 'test/input2/'+ input2_labels[i]
            label_out.append(input1_labels[i])
            input1_tif=tiff.imread(Input1_Path) 
            input2_tif=tiff.imread(Input2_Path) 
            inmin1, inmax1 = input1_tif.min(), input1_tif.max()
            normal_input1_tif=(input1_tif-inmin1)/(inmax1-inmin1)        
            inmin2, inmax2 = input2_tif.min(), input2_tif.max()
            normal_input2_tif=(input2_tif-inmin2)/(inmax2-inmin2)
            ##x_data = np.append(x_data, output)  
            x1_data = np.append(x1_data, normal_input1_tif) 
            x2_data = np.append(x2_data, normal_input2_tif) 
        if end>=maxlen:
            end=maxlen
            batch_index = 0
        else:
            batch_index += batch_size  # update index for the next batch
        x1_data_ = x1_data.reshape(batch_size, input_data_depth * input_data_height * input_data_width )
        x2_data_ = x2_data.reshape(batch_size, input_data_depth * input_data_height * input_data_width )
        return x1_data_,x2_data_,label_out


class DenseDeconNet:
    def __init__(self):
        print('New DenseDeconNet Network')
        self.input_image1 = {}
        self.input_image2 = {}
        self.ground_truth = {}
        self.cast_image1= None
        self.cast_image2 = None
        self.cast_image= None
        self.cast_ground_truth = None
        self.is_traing =None
        self.m=None
        self.loss, self.loss_square, self.loss_all, self.train_step = [None] * 4
        self.prediction, self.correct_prediction, self.accuracy = [None] * 3
        self.result_conv = {}
        self.result_relu = {}
        self.result_from_contract_layer = {}
        self.w = {}
        self.sub_diff,self.mse_square,self.mse=[None] * 3
        self.mean_prediction,self.mean_gt=[None] * 2
        self.sigma_prediction,self.sigma_gt,self.sigma_cross=[None] * 3
        self.SSIM_1,self.SSIM_2,self.SSIM_3,self.SSIM_4,self.SSIM=[None] * 5
        self.learning_rate=[None]  
        self.prediction_min= [None]     
        
    def init_w(self, shape, name):
        with tf.name_scope('init_w'):
            stddev = 0.1
            w = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32), name=name)
            return w
        
    @staticmethod
    def init_b(shape, name):
        with tf.name_scope('init_b'):
            return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name=name)
        
        
    ############### batch normalization  ###################
    @staticmethod
    def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, name='BatchNorm3d'):       
        with tf.variable_scope(name):
            params_shape = x.shape[-1:]
            beta = tf.Variable(tf.constant(0.0, shape=params_shape), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=params_shape), name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
            ema = tf.train.ExponentialMovingAverage(decay)
            
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
    
            mean, variance = tf.cond(tf.equal(is_training,True), mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
            

            normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=eps)
        return normed
        
    @staticmethod
    def copy_and_crop_and_merge(result_from_downsampling, result_from_upsampling):
        return tf.concat(values=[result_from_downsampling, result_from_upsampling], axis=-1)##axis=4
    
    
    def set_up_net(self, batch_size):
        # input
        with tf.name_scope('input'):
            self.input_image1 = tf.placeholder(
                    dtype=tf.float32, shape=[batch_size, input_data_depth*input_data_height*input_data_width*input_data_channel], name='input_images'
                    )
            self.input_image2 = tf.placeholder(
                    dtype=tf.float32, shape=[batch_size, input_data_depth*input_data_height*input_data_width*input_data_channel], name='input_images'
                    )
            self.cast_image1 = tf.reshape(
                    tensor=self.input_image1,
                    shape=[batch_size, input_data_depth,input_data_height,input_data_width,input_data_channel]
                    )
            self.cast_image2 = tf.reshape(
                    tensor=self.input_image2,
                    shape=[batch_size, input_data_depth,input_data_height,input_data_width,input_data_channel]
                    )
            self.cast_image = tf.concat([self.cast_image1,self.cast_image2],4)
            
            self.ground_truth = tf.placeholder(
                    dtype=tf.float32, shape=[batch_size, output_data_depth*output_data_height*output_data_width*output_data_channel], name='input_images'
                    )
            self.cast_ground_truth = tf.reshape(
                    tensor=self.ground_truth,
                    shape=[batch_size, output_data_depth,output_data_height,output_data_width,output_data_channel]
                    )
            
            self.is_traing = tf.placeholder(tf.bool)
            
            

            
        # layer 1
        with tf.name_scope('layer_1'):
            # conv_1
            self.w[1] = self.init_w(shape=[3, 3, 3, 2, 4], name='w_1')
            result_conv_1 = tf.nn.conv3d(
                    input=self.cast_image, filter=self.w[1],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_1 = self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_1 = tf.nn.relu(normed_batch_1, name='relu_1')
            
            
            self.w[2] = self.init_w(shape=[3,3, 3, 4, 8], name='w_1')
            result_conv_2 = tf.nn.conv3d(
                    input=result_prelu_1, filter=self.w[2],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_2 = self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_2 = tf.nn.relu(normed_batch_2, name='relu_1')

            
            result_merge_B1= self.copy_and_crop_and_merge(
                    result_from_downsampling=result_prelu_1, result_from_upsampling=result_prelu_2)
            
            self.w[3] = self.init_w(shape=[3, 3, 3, 12, 4], name='w_1')
            result_conv_3 = tf.nn.conv3d(
                    input=result_merge_B1, filter=self.w[3],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_3 = self.batch_norm(x=result_conv_3, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_3 = tf.nn.relu(normed_batch_3, name='relu_1')
            
            result_merge_B2_1= self.copy_and_crop_and_merge(
                    result_from_downsampling=result_prelu_1, result_from_upsampling=result_prelu_3)
            result_merge_B2= self.copy_and_crop_and_merge(
                    result_from_downsampling=result_prelu_2, result_from_upsampling=result_merge_B2_1)

            self.w[4] = self.init_w(shape=[3, 3, 3, 16, 4], name='w_1')            
            result_conv_4 = tf.nn.conv3d(
                    input=result_merge_B2, filter=self.w[4],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_4 = self.batch_norm(x=result_conv_4, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_4 = tf.nn.relu(normed_batch_4, name='relu_1')

##############################   down_sampleing   ###################################################################

            self.w[5] = self.init_w(shape=[2, 2, 2, 4, 8], name='w_1')             
            result_conv_5 = tf.nn.conv3d(
                    input=result_prelu_4, filter=self.w[5],
                    strides=[1, 2, 2, 2, 1], padding='VALID', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_5 = self.batch_norm(x=result_conv_5, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_5 = tf.nn.relu(normed_batch_5, name='relu_1')
            
            self.w[6] = self.init_w(shape=[3, 3, 3, 8, 4], name='w_1')
            result_conv_6 = tf.nn.conv3d(
                    input=result_prelu_5, filter=self.w[6],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_6 = self.batch_norm(x=result_conv_6, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_6 = tf.nn.relu(normed_batch_6, name='relu_1')

            result_merge_B3= self.copy_and_crop_and_merge(
                    result_from_downsampling=result_prelu_5, result_from_upsampling=result_prelu_6)
            
            self.w[7] = self.init_w(shape=[3, 3, 3, 12, 4], name='w_1')
            result_conv_7 = tf.nn.conv3d(
                    input=result_merge_B3, filter=self.w[7],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_7 = self.batch_norm(x=result_conv_7, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_7 = tf.nn.relu(normed_batch_7, name='relu_1')
            
            result_merge_B4_1= self.copy_and_crop_and_merge(
                    result_from_downsampling=result_prelu_7, result_from_upsampling=result_prelu_5)
            result_merge_B4= self.copy_and_crop_and_merge(
                    result_from_downsampling=result_prelu_6, result_from_upsampling=result_merge_B4_1)

            self.w[8] = self.init_w(shape=[3, 3, 3, 16, 8], name='w_1')            
            result_conv_8 = tf.nn.conv3d(
                    input=result_merge_B4, filter=self.w[8],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_8 = self.batch_norm(x=result_conv_8, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_8 = tf.nn.relu(normed_batch_8, name='relu_1')
            
########################################################################################################################
            self.w[9] = self.init_w(shape=[1, 1, 1, 8, 4], name='w_1')             
            result_conv_9 = tf.nn.conv3d(
                    input=result_prelu_8, filter=self.w[9],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_9 = self.batch_norm(x=result_conv_9, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_9 = tf.nn.relu(normed_batch_9, name='relu_1')


            self.w[10] = self.init_w(shape=[3, 3, 3, 4, 8], name='w_1')
            result_conv_10 = tf.nn.conv3d(
                    input=result_prelu_9, filter=self.w[10],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_10 = self.batch_norm(x=result_conv_10, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_10 = tf.nn.relu(normed_batch_10, name='relu_1')

            result_merge_B5= self.copy_and_crop_and_merge(
                    result_from_downsampling=result_prelu_9, result_from_upsampling=result_prelu_10)
            
            self.w[11] = self.init_w(shape=[3, 3, 3, 12, 4], name='w_1')
            result_conv_11 = tf.nn.conv3d(
                    input=result_merge_B5, filter=self.w[11],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_11 = self.batch_norm(x=result_conv_11, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_11 = tf.nn.relu(normed_batch_11, name='relu_1')
            
            result_merge_B6_1= self.copy_and_crop_and_merge(
                    result_from_downsampling=result_prelu_11, result_from_upsampling=result_prelu_9)
            result_merge_B6= self.copy_and_crop_and_merge(
                    result_from_downsampling=result_prelu_10, result_from_upsampling=result_merge_B6_1)

            self.w[12] = self.init_w(shape=[3, 3, 3, 16, 8], name='w_1')            
            result_conv_12 = tf.nn.conv3d(
                    input=result_merge_B6, filter=self.w[12],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_12 = self.batch_norm(x=result_conv_12, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_12 = tf.nn.relu(normed_batch_12, name='relu_1')
            
###############################up_sampling ###########################################################                        
            self.w[13] = self.init_w(shape=[2, 2, 2, 4, 8], name='w_1')
            result_conv_15 = tf.nn.conv3d_transpose(
                    value=result_prelu_12, filter=self.w[13],
                    output_shape=[batch_size,input_data_depth, input_data_height, input_data_width,4],
                    strides=[1, 2, 2, 2, 1], padding='VALID', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_15 = self.batch_norm(x=result_conv_15, is_training=self.is_traing, name='layer_1_conv_1')
            add_layer=tf.add(normed_batch_15,result_prelu_1)
            result_prelu_15 = tf.nn.relu(add_layer, name='relu_1')

            
######################################################################################################################
            


            self.w[15] = self.init_w(shape=[3, 3, 3, 4, 1], name='w_1')
            result_conv_18 = tf.nn.conv3d(
                    input=result_prelu_15, filter=self.w[15],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            #result_conv_bias_1=tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias')
            normed_batch_18 = self.batch_norm(x=result_conv_18, is_training=self.is_traing, name='layer_1_conv_1')
            result_prelu_18 = tf.nn.leaky_relu(normed_batch_18, name='relu_1')

                        

            self.prediction = result_prelu_18
            self.prediction_min=tf.reduce_min(self.prediction)

############################### loss function part #################################            
        with tf.name_scope('MSE'):
            self.sub_diff =self.prediction-self.cast_ground_truth
            self.mse_square =tf.square(self.sub_diff)
            self.mse = tf.reduce_mean(self.mse_square)

        with tf.name_scope('SSIM'):
            self.mean_prediction =tf.reduce_mean(self.prediction)
            self.mean_gt =tf.reduce_mean(self.cast_ground_truth)
            self.sigma_prediction=tf.reduce_mean(tf.square(tf.subtract(self.prediction,self.mean_prediction)))
            self.sigma_gt=tf.reduce_mean(tf.square(tf.subtract(self.cast_ground_truth,self.mean_gt)))
            self.sigma_cross=tf.reduce_mean(tf.multiply(tf.subtract(self.prediction,self.mean_prediction),
                                                        tf.subtract(self.cast_ground_truth,self.mean_gt)))
            self.SSIM_1=2*tf.multiply(self.mean_prediction,self.mean_gt)+1e-4
            self.SSIM_2=2*self.sigma_cross+9e-4
            self.SSIM_3=tf.square(self.mean_prediction)+tf.square(self.mean_gt)+1e-4
            self.SSIM_4=self.sigma_prediction+self.sigma_gt+9e-4
            self.SSIM=tf.div(tf.multiply(self.SSIM_1,self.SSIM_2),tf.multiply(self.SSIM_3,self.SSIM_4))
            
        with tf.name_scope('loss'):
            #self.loss =0.4*self.mse-0.6*tf.log((1+self.SSIM)/2)
            self.loss =self.mse-1.1*tf.log((1+self.SSIM)/2)-1.3*self.prediction_min #1.25

######################### learning rate ############################
        # Gradient Descent        
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.004,self.global_step,600,0.96,staircase=False) #previous 0.04,200,0.98#0.005,200 
        with tf.name_scope('Gradient_Descent'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)



          
            
    def training(self):
        train_dir=data_dir+'model/' #save logs files for training process
        checkpoint_path=os.path.join(train_dir,'model.ckpt')
        pre_parameters_saver = tf.train.Saver() #sav
        all_parameters_saver = tf.train.Saver() #save
        with tf.Session() as sess:  # 开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
#            all_parameters_saver.restore(sess=sess, save_path=checkpoint_path)
            ##start training
            sum_los = 0.0
            time_start=time.time()
            for k in range(train_iter_num):
                train_images1,train_images2,train_GT,label2out=get_data_tiff(mode,train_batch_size)
                mse,ssim,lo= sess.run([self.mse,self.SSIM,self.loss],
                                         feed_dict={self.input_image1:train_images1,self.input_image2:train_images2, self.ground_truth: train_GT,self.is_traing: True}
                                         )
                sum_los += lo               
                if k % 100 == 0:
                    time_end=time.time()
                    print('num %d, mse: %.6f, SSIM: %.6f, loss: %.6f, runtime:%.6f ' % (k, mse, ssim, lo,time_end-time_start))
                sess.run([self.train_step],
                         feed_dict={self.input_image1:train_images1,self.input_image2:train_images2, self.ground_truth: train_GT,self.is_traing: True}
                         )
#  show sumloss of train, number(k+1)%(how many data we have)
                if (k+1)%150==0:
                    print('sum_lo: %.6f' %(sum_los))
                    sum_los = 0.0
# train result save part
                if (k+1)%200==0:
                    image= sess.run([self.prediction],
                                      feed_dict={self.input_image1:train_images1,self.input_image2:train_images2, self.ground_truth: train_GT,self.is_traing: True})
                    image1=np.array(image)
                    reshape_image=image1.reshape(train_batch_size,output_data_depth,output_data_height,output_data_width)
                    print(label2out)
                    for v in range(train_batch_size):
                        single=reshape_image[v]
                        filenames_out=data_dir+'output/'+str(k)+'_'+label2out[v]
                        tiff.imsave(filenames_out,single)
#                        tiff.imsave(filenames_gt,single_gt)
# set when the model parameter save
                if (k+1)%2000== 0:
                    all_parameters_saver.save(sess=sess, save_path=checkpoint_path)
            print("Done training")
        sess.close()
        
        
    def validating(self):
        train_dir=data_dir+'model/' #save logs files for training process
        checkpoint_path=os.path.join(train_dir,'model.ckpt')
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            all_parameters_saver.restore(sess=sess, save_path=checkpoint_path)
            sum_los = 0.0
            for m in range(val_iter_num):
                time_start=time.time()
                test_images1,test_images2,test_GT,label2out=get_data_tiff(mode,val_batch_size)
                image, mse, ssim, los = sess.run([self.prediction, self.mse, self.SSIM, self.loss],
                                      feed_dict={self.input_image1:test_images1, self.input_image2:test_images2, self.ground_truth: test_GT,self.is_traing:True}
                                      )
                sum_los += los
                image1=np.array(image)
                reshape_image=image1.reshape(test_batch_size,output_data_depth,output_data_height,output_data_width)
                for v in range(val_batch_size):
                    single=reshape_image[v]
                    filenames_out=data_dir+'output/'+str(m)+'_'+label2out[v]
                    tiff.imsave(filenames_out,single)
                #########  save output image #####################
                if m % 1 == 0:
                    time_end=time.time()
                    print('num %d, mse: %.6f, SSIM: %.6f, loss: %.6f, runtime:%.6f ' % (m, mse, ssim, los,time_end-time_start))
        print('Done testing')

    def testing(self):
        train_dir=data_dir+'model/' #save logs files for training process
        checkpoint_path=os.path.join(train_dir,'model.ckpt')
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  
#            time_start=time.time()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            all_parameters_saver.restore(sess=sess, save_path=checkpoint_path)
            for m in range(test_iter_num):
                time_start=time.time()
                test_images1,test_images2, label2out=get_data_tiff(mode,test_batch_size)
                image= sess.run([self.prediction],feed_dict={self.input_image1:test_images1, self.input_image2:test_images2, self.is_traing:True})
                image1=np.array(image)
                reshape_image=image1.reshape(test_batch_size,output_data_depth,output_data_height,output_data_width)
                for v in range(test_batch_size):
                    single=reshape_image[v]
                    filenames_out=data_dir+'output/'+'_'+label2out[v]
                    tiff.imsave(filenames_out,single)
                #########  save output image #####################
                if m % 1 == 0:
                    time_end=time.time()
                    print('num %d, runtime:%.6f ' % (m,time_end-time_start))
        print('Done testing')
        
        
def main():
    net = DenseDeconNet()
    if mode=='training':
        net.set_up_net(train_batch_size)
        net.training()
    elif mode=='validating':
        net.set_up_net(test_batch_size)
        net.validating()
    elif mode=='testing':
        net.set_up_net(test_batch_size)
        net.testing()                
    else: print('wrong mode!')

main()
 

                      #########  to be continued ########################            
