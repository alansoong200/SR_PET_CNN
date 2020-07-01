#for 2 channels
import os
import csv
from PIL import Image
from PIL import ImageFilter
import pandas as pd
import numpy as np
import random
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import scipy.io
#import alexnet as a
from scipy.misc import toimage
from torchsample.transforms import *
from sklearn.metrics import confusion_matrix
import glob
from torchvision.datasets.folder import IMG_EXTENSIONS
import time
IMG_EXTENSIONS.append('tif')
import pytorch_ssim
# Hyper Parameters
EPOCH = 1             # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1
rootdir_t = './data_t_20s_d_1/'
rootdir_v = './data_v_15/'
STEP = 310

MSE_T = np.zeros(EPOCH)
MSE_V = np.zeros(EPOCH)

dir_name = "PAPER_SRCNN_4ch_l1_20s_d_1"

path = "./params_adni/" + dir_name
#output_path = "./outputs_adni/" + dir_name
output_path = "./Revision_imgs/s4/"


v_MSE_name = "./MSE/" + dir_name + ".txt"
crop_size = 32#96

outsave=1
psave=1

in_ch = 4

#data_t l1:
ep = [150, 149, 150, 150]

#data_t l1 cut image 31-120:

ep1 = 150

ll = 90
kk = 90

ch = 64
ksize = 3
pad = 1

res_rate = 0.05
#SRCNN method
class SRCNN(nn.Module):
	def __init__(self):
		super(SRCNN, self).__init__()
		self.input = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=in_ch,              # input height
                		out_channels=ch,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step
                		padding=pad,                  
            			),
	    		
 	    		nn.LeakyReLU(inplace=False),
            
			)
		self.layers = nn.Sequential(
		
                	nn.Conv2d(in_channels=ch, out_channels=ch/2, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),


		)

		self.output = nn.Sequential(
            		nn.Conv2d(
                		in_channels=ch/2,              # input height
                		out_channels=1,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step
                		padding=pad,
            		),
            		
        
        	)



            

		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
                		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                		m.weight.data.normal_(0, math.sqrt(2. / n))
	

    	def I_weights(self, m):
        	#print(m)
        	if type(m) == nn.Conv2d:
            		w = 1
            	if m.in_channels == m.out_channels and m.kernel_size > 2:
               		w = 3
            	alpha = math.pow(m.kernel_size[0]*1.0/m.stride[0]/w, 2.0)
            	scale = math.sqrt(6.0/(m.in_channels+m.out_channels*alpha))
            	#print scale
            	m.weight.data.uniform_(-scale,scale)
            	#print(m.weight) 
   
    	"""
   	def inititalize_W(self):
        #self.I_weights(self.conv1[0])
        #self.I_weights(self.conv2[0])
        #self.I_weights(self.conv3[0])
    	"""


    	def forward(self, x):
		LR = x[:,:1,:,:]
		x1 = self.input(x)
		x2 = self.layers(x1)
		x3 = self.output(x2)
		
		out = torch.add(x3, LR)
		
		return out

    	def save_p(self, epoch):
		modelparam = path + "/inputEpoch" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam)    
		modelparam = path + "/layersEpoch" + str(epoch) +".pkl"
	        torch.save(self.layers.state_dict(), modelparam)
		modelparam = path + "/outputEpoch" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam(self, epoch):	
		modelparam = path + "/inputEpoch" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = path + "/layersEpoch" + str(epoch) +".pkl"
	        self.layers.load_state_dict(torch.load(modelparam))
	        modelparam = path + "/outputEpoch" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"




def PSNR(imgs, img_y):
    
    	imgs = imgs.clamp(0, 1)

    	out = imgs - img_y
    
    	MSE = np.mean(out.numpy() ** 2)
    	#print "MSE : ", MSE
    	if MSE < 0:
        
		print "something wrong!!"
		quit()

    	#return 10 * math.log10(1.0 / MSE), MSE


    	return MSE


def rand_crop_loader(path):
	

	#print path
	data = scipy.io.loadmat(path)

	imgs = np.squeeze(data['mat'])
	p_max = data['pmax']
	m_max = data['mmax']
	Rrange = 179.6051
	Rmin = 0.7071
	#print im.size
        imgs_1 = np.zeros((256,256,4))
	imgs_2 = np.zeros((256,256,3))


	if in_ch != 3:
		temp = imgs[:,:,5] / p_max[0]
		imgs[:,:,0] = imgs[:,:,0] / p_max[0]
		imgs[:,:,1] = imgs[:,:,1] / m_max[0]
		imgs[:,:,2] = (imgs[:,:,3] - Rmin) / Rrange 
		imgs[:,:,3] = (imgs[:,:,4] + 127.5) / 255.0
		imgs[:,:,4] = temp

	else:
		temp = imgs[:,:,5] / p_max[0]
		imgs[:,:,0] = imgs[:,:,0] / p_max[0]
		#imgs[:,:,1] = imgs[:,:,1] / m_max[0]
		imgs[:,:,1] = (imgs[:,:,3] - Rmin) / Rrange 
		imgs[:,:,2] = (imgs[:,:,4] + 127.5) / 255.0
		imgs[:,:,4] = temp


	

	#img1 = toimage(imgs[:,:,2])

	#img1.show()	
	#quit()
	x = random.randint(40, 180)#126
	y = random.randint(25, 198)#126
	#return imgs[random.randint(0,127), random.randint(0,127), :], p_max[0]
	return imgs[x:x+crop_size, y:y+crop_size,:], p_max[0]


def default_loader(path):
	

	#print path
	data = scipy.io.loadmat(path)
	#print(path)
	imgs = np.squeeze(data['mat'])
	p_max = data['pmax']
	m_max = data['mmax']
	Rrange = 179.6051
	Rmin = 0.7071
	#print im.size
        #new_imgs = np.zeros((256,256,3))

	if in_ch != 3:
		temp = imgs[:,:,2] / p_max[0]
		imgs[:,:,0] = imgs[:,:,0] / p_max[0]
		imgs[:,:,1] = imgs[:,:,1] / m_max[0]
		imgs[:,:,2] = (imgs[:,:,3] - Rmin) / Rrange 
		imgs[:,:,3] = (imgs[:,:,4] + 127.5) / 255.0
		imgs[:,:,4] = temp

	else:
		temp = imgs[:,:,2] / p_max[0]
		imgs[:,:,0] = imgs[:,:,0] / p_max[0]
		#imgs[:,:,1] = imgs[:,:,1] / m_max[0]
		imgs[:,:,1] = (imgs[:,:,3] - Rmin) / Rrange 
		imgs[:,:,2] = (imgs[:,:,4] + 127.5) / 255.0
		imgs[:,:,4] = temp



	name = os.path.basename(path)
	#img1 = toimage(imgs[:,:,2])

	#img1.show()	
	#quit()

	#print (imgs.dtype)
	#quit()
	return imgs, p_max[0], name


class myImageFloder(Data.Dataset):
    	def __init__(self, root, label, transform = None, target_transform=None, loader=default_loader, mode = 0):
        
		self.root = root
		if mode == 0:
        		self.imgs = glob.glob(root + "p2*.mat")
		elif mode == 1:
			self.imgs = glob.glob(root + "p8*.mat")
		elif mode == 2:
			self.imgs = glob.glob(root + "p10*.mat")
		elif mode == 3:
			self.imgs = glob.glob(root + "p13*.mat")
		elif mode == 4:
			self.imgs = glob.glob(root + "p16*.mat")
		elif mode == 5:
			self.imgs = glob.glob(root + "p21*.mat")
		elif mode == 6:
			self.imgs = glob.glob(root + "p22*.mat")
		elif mode == 7:
			self.imgs = glob.glob(root + "p23*.mat")
		elif mode == 8:
			self.imgs = glob.glob(root + "p24*.mat")
		elif mode == 9:
			self.imgs = glob.glob(root + "p25*.mat")
		elif mode == 10:
			self.imgs = glob.glob(root + "p26*.mat")
		elif mode == 11:
			self.imgs = glob.glob(root + "p27*.mat")
		elif mode == 12:
			self.imgs = glob.glob(root + "p28*.mat")
		elif mode == 13:
			self.imgs = glob.glob(root + "p29*.mat")
		else:
			self.imgs = glob.glob(root + "p30*.mat")
        	#self.classes = class_names
        	self.transform = transform
        	self.target_transform = target_transform
        	self.loader = loader

    	def __getitem__(self, index):
        	img_path = self.imgs[index]
        	img1, pmax, name = self.loader(img_path)

		img1 = torch.FloatTensor(torch.from_numpy(img1.transpose((2, 0, 1))).float())

        	if self.transform is not None:

       			img1[:4,:,:]= self.transform(img1[:4,:,:])
		


		#img = torch.from_numpy(img.transpose((2, 0, 1))).float()

		
        
			
		return img1, pmax, name

    	def __len__(self):
        	return len(self.imgs)
    
    	def getName(self):
        	return self.classes





#t_cnn = ss()
#t_cnn.restore_netparam(99)
#t_cnn.cuda()

cnn = SRCNN()
#cnn.inititalize_W()
cnn.restore_netparam(ep1)
cnn.cuda()	
print(cnn)  # net architecture



val_2 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 0),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)

val_8 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 1),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)


val_10 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 2),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)


val_13 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 3),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)


val_16 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 4),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)

val_21 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 5),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)

val_22 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 6),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)
val_23 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 7),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)

val_24 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 8),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)

val_25 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 9),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)

val_26 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 10),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)

val_27 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 11),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)

val_28 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 12),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)

val_29 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 13),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)

val_30 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 14),
	batch_size= 1, 
	shuffle= False, 
	num_workers=1, 
	drop_last = False)





#loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
loss_func = nn.MSELoss()
loss_func1 = nn.L1Loss()

print "SRCNN true pet" 
print "Epoch : ", EPOCH, " Batch Size : ", BATCH_SIZE
print "Input channels : ", in_ch
print "rootdir_t : ", rootdir_t
print "rootdir_v : ", rootdir_v
print "Path : ", path







#optimizer = torch.optim.SGD(cnn.parameters(), lr=0.00001)
for epoch in range(EPOCH):


	"""			
	temp2 = 0.0
	ssim2 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax) in enumerate(val_2):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp2 += loss.data[0]
        		ssim2 += pytorch_ssim.ssim(v_output, v_y)

        		res = np.squeeze(v_output[0].cpu().data.numpy()) * pmax
			name = output_path + "p2.mat"
			scipy.io.savemat(name, {'img': res.numpy()})

	print "Val loss (2): ", temp2
	psnr2 = 10 * math.log10(1.0 / (temp2/ll))
	print "psnr (2): ", 10 * math.log10(1.0 / (temp2/ll))
	ssim2 = ssim2/kk	
	print "ssim (2): ", ssim2
	"""
	temp8 = 0.0
	ssim8 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_8):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)
	

			temp8 += loss.data[0]
			ssim8 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy()) 
			#name = output_path + "p8.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

        
	print "Val loss (8): ", temp8
	psnr8 = 10 * math.log10(1.0 / (temp8/ll))
	print "psnr (8): ", 10 * math.log10(1.0 / (temp8/ll))
	ssim8 = ssim8/kk	
	print "ssim (8): ", ssim8



	temp10 = 0.0
	ssim10 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_10):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp10 += loss.data[0]
        		ssim10 += pytorch_ssim.ssim(v_output, v_y)        		
			res = np.squeeze(v_output[0].cpu().data.numpy())
			#name = output_path + "p10.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})


	print "Val loss (10): ", temp10
	psnr10 = 10 * math.log10(1.0 / (temp10/ll))
	print "psnr (10): ", 10 * math.log10(1.0 / (temp10/ll))
	ssim10= ssim10/kk	
	print "ssim (10): ", ssim10


	temp13 = 0.0
	ssim13 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_13):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp13 += loss.data[0]
                	ssim13 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy()) 
			#name = output_path + "p13.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (13): ", temp13
	psnr13 = 10 * math.log10(1.0 / (temp13/ll))
	print "psnr (13): ", 10 * math.log10(1.0 / (temp13/ll))
	ssim13 = ssim13/kk	
	print "ssim (13): ", ssim13

	temp16 = 0.0
	ssim16 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_16):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp16 += loss.data[0]
                        ssim16 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy()) 
			#name = output_path + "p16.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (16): ", temp16
	psnr16 = 10 * math.log10(1.0 / (temp16/ll))
	print "psnr (16): ", 10 * math.log10(1.0 / (temp16/ll))
	ssim16= ssim16/kk	
	print "ssim (16): ", ssim16


	print "------------------------------"



	temp21 = 0.0
	ssim21 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_21):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp21 += loss.data[0]
                        ssim21 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy()) 
			#name = output_path + "p21.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (21): ", temp21
	psnr21 = 10 * math.log10(1.0 / (temp21/ll))
	print "psnr (21): ", 10 * math.log10(1.0 / (temp21/ll))
	ssim21= ssim21/kk	
	print "ssim (21): ", ssim21


	temp22 = 0.0
	ssim22 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_22):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp22 += loss.data[0]
                        ssim22 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy())
			#name = output_path + "p22.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (22): ", temp22
	psnr22 = 10 * math.log10(1.0 / (temp22/ll))
	print "psnr (22): ", 10 * math.log10(1.0 / (temp22/ll))
	ssim22= ssim22/kk	
	print "ssim (22): ", ssim22



	temp23 = 0.0
	ssim23 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_23):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp23 += loss.data[0]
                        ssim23 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy()) 
			#name = output_path + "p23.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (23): ", temp23
	psnr23 = 10 * math.log10(1.0 / (temp23/ll))
	print "psnr (23): ", 10 * math.log10(1.0 / (temp23/ll))
	ssim23 = ssim23/kk	
	print "ssim (23): ", ssim23




	temp24 = 0.0
	ssim24 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_24):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp24 += loss.data[0]
                        ssim24 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy())
			#name = output_path + "p24.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (24): ", temp24
	psnr24 = 10 * math.log10(1.0 / (temp24/ll))
	print "psnr (24): ", 10 * math.log10(1.0 / (temp24/ll))
	ssim24 = ssim24/kk	
	print "ssim (24): ", ssim24


	temp25 = 0.0
	ssim25 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_25):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp25 += loss.data[0]
                        ssim25 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy()) 
			#name = output_path + "p25.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (25): ", temp25
	psnr25 = 10 * math.log10(1.0 / (temp25/ll))
	print "psnr (25): ", 10 * math.log10(1.0 / (temp25/ll))
	ssim25= ssim25/kk	
	print "ssim (25): ", ssim25


	temp26 = 0.0
	ssim26 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_26):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp26 += loss.data[0]
                        ssim26 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy())
			#name = output_path + "p26.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (26): ", temp26
	psnr26 = 10 * math.log10(1.0 / (temp26/ll))
	print "psnr (26): ", 10 * math.log10(1.0 / (temp26/ll))
	ssim26= ssim26/kk	
	print "ssim (26): ", ssim26


	temp27 = 0.0
	ssim27 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_27):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp27 += loss.data[0]
                        ssim27 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy()) 
			#name = output_path + "p27.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (27): ", temp27
	psnr27 = 10 * math.log10(1.0 / (temp27/ll))
	print "psnr (27): ", 10 * math.log10(1.0 / (temp27/ll))
	ssim27= ssim27/kk	
	print "ssim (27): ", ssim27



	temp28 = 0.0
	ssim28 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_28):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp28 += loss.data[0]
                        ssim28 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy())
			#name = output_path + "p28.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (28): ", temp28
	psnr28 = 10 * math.log10(1.0 / (temp28/ll))
	print "psnr (28): ", 10 * math.log10(1.0 / (temp28/ll))
	ssim28= ssim28/kk	
	print "ssim (28): ", ssim28



	temp29 = 0.0
	ssim29 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_29):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp29 += loss.data[0]
                        ssim29 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy())
			#name = output_path + "p29.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})
	print "Val loss (29): ", temp29
	psnr29 = 10 * math.log10(1.0 / (temp29/ll))
	print "psnr (29): ", 10 * math.log10(1.0 / (temp29/ll))
	ssim29 = ssim29/kk	
	print "ssim (29): ", ssim29


	temp30 = 0.0
	ssim30 = 0.0
	print "------------------------------"
	
	
	#validation
	for step, (x, pmax, name) in enumerate(val_30):
		
		with torch.no_grad():
			#print x.size()
               	 	temp_x = x.size()[2]
                	temp_y = x.size()[3]


			out = torch.FloatTensor(1, in_ch, temp_x, temp_y)
               	 	out1 = torch.FloatTensor(1, 1, temp_x, temp_y)
                	out[:,:,:,:] =  x[:,:in_ch,:,:]
                	out1[:,0,:,:] = x[:,4,:,:]


                	v_input = Variable(out.cuda())
                	v_y = Variable(out1.cuda())

                	v_output = cnn(v_input)
	
                	loss = loss_func(v_output, v_y)

			temp30 += loss.data[0]
                        ssim30 += pytorch_ssim.ssim(v_output, v_y)
        		res = np.squeeze(v_output[0].cpu().data.numpy())
			#name = output_path + "p30.mat"
			n = ''.join(name)
			loc = output_path + n
			scipy.io.savemat(loc, {'img': res})

	print "Val loss (30): ", temp30
	psnr30 = 10 * math.log10(1.0 / (temp30/ll))
	print "psnr (30): ", 10 * math.log10(1.0 / (temp30/ll))
	ssim30 = ssim30/kk	
	print "ssim (30): ", ssim30





	print "------------------------------"


	print "Epoch: ", ep1

	print "ll: ", ll

	print "kk: ", kk
 
	print "The PSNR Max is: ", max( psnr8, psnr10, psnr13, psnr16, psnr21, psnr22, psnr23, psnr24, psnr25, psnr26, psnr27, psnr28, psnr29, psnr30)
 	print "The PSNR Min is: ", min( psnr8, psnr10, psnr13, psnr16, psnr21, psnr22, psnr23, psnr24, psnr25, psnr26, psnr27, psnr28, psnr29, psnr30)
	print "The PSNR Avg is: ", sum([psnr8, psnr10, psnr13, psnr16, psnr21, psnr22, psnr23, psnr24, psnr25, psnr26, psnr27, psnr28, psnr29, psnr30]) / len([psnr8, psnr10, psnr13, psnr16, psnr21, psnr22, psnr23, psnr24, psnr25, psnr26, psnr27, psnr28, psnr29, psnr30]) 
	print "The PSNR STD is: ", np.std([psnr8, psnr10, psnr13, psnr16, psnr21, psnr22, psnr23, psnr24, psnr25, psnr26, psnr27, psnr28, psnr29, psnr30], axis=0)
	print "The SSIM Max is: ", max( ssim8, ssim10, ssim13, ssim16, ssim21, ssim22, ssim23, ssim24, ssim25, ssim26, ssim27, ssim28, ssim29, ssim30)
 	print "The SSIM Min is: ", min( ssim8, ssim10, ssim13, ssim16, ssim21, ssim22, ssim23, ssim24, ssim25, ssim26, ssim27, ssim28, ssim29, ssim30)
	print "The SSIM Avg is: ", sum([ssim8, ssim10, ssim13, ssim16, ssim21, ssim22, ssim23, ssim24, ssim25, ssim26, ssim27, ssim28, ssim29, ssim30]) / len([ssim8, ssim10, ssim13, ssim16, ssim21, ssim22, ssim23, ssim24, ssim25, ssim26, ssim27, ssim28, ssim29, ssim30]) 
	print "The SSIM STD is: ", np.std([ssim8, ssim10, ssim13, ssim16, ssim21, ssim22, ssim23, ssim24, ssim25, ssim26, ssim27, ssim28, ssim29, ssim30], axis=0)	

print "Finished"



