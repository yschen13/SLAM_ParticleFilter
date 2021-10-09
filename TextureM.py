import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot  as plt
import matplotlib.cm as cm
import os 
from map_utils import bresenham2D, mapCorrelation
import random
from scipy import ndimage
from PIL import Image

dataset = 20
with np.load("Kinect%d.npz"%dataset) as data:
	disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
	rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

with np.load("Encoders%d.npz"%dataset) as data:
	encoder_stamps = data["time_stamps"] # encoder time stamps
	encoder_counts = data["counts"] # 4 x n encoder counts

N= 10
lamda = np.load('Set'+str(dataset)+'_N'+str(N)+'_NStation_lamda.npy')
CurrentBest = np.load('Set'+str(dataset)+'_N'+str(N)+'_NStation_CurrentBest.npy')

# Threshold a good map
# map_thres = lamda<(-800)
# plt.subplot(1,2,1)
# plt.imshow(lamda)
# plt.subplot(1,2,2)
# plt.imshow(map_thres)
# plt.savefig('Map_thres.png',dpi=500)
# plt.close()


xmin=-55.0;ymin=-55.0;res=0.1 # meters/grid
thres = 0.4 # height threshold
CurrentBest = CurrentBest[:,None]
color_grid = np.zeros((1100,1100,3))
color_grid_N = np.zeros((1100,1100))



start_t = np.where(encoder_counts[0,:]!=0)[0][0]
for t in np.arange(start_t, encoder_stamps.shape[0]-1):
# for t in np.arange(1622, encoder_stamps.shape[0]-1):
	roll=0; pitch=0.36; yaw=0.021+CurrentBest[2,:,t][0] # angles of d.camera in world frame
	R1 = np.array(([np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]))
	R2 = np.array(([np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]))
	R3 = np.array(([1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]))
	Rwc = np.matmul(np.matmul(R1,R2),R3)
	pwr = np.array([CurrentBest[0,:,t][0],CurrentBest[1,:,t][0],0.514])
	prc = np.array([0.18,0.005,0.36])-np.array([(330-150)/2000,0,0.514])
	pwc = pwr + prc
	Roc = np.array(([0,-1,0],[0,0,-1],[1,0,0]))
	K = np.array(([585.05108211,0,242.94140713],[0, 585.05108211, 315.83800193],[0,0,1]))
	invK = np.linalg.inv(K)
	CurrentTime = encoder_stamps[t]
	disp_idx = np.argmin(np.abs(disp_stamps-CurrentTime))
	rgb_idx = np.argmin(np.abs(rgb_stamps-CurrentTime))
	img = Image.open('dataRGBD/Disparity'+str(dataset)+'/disparity'+str(dataset)+'_'+str(disp_idx+1)+'.png')
	disp = np.array(img.getdata(),np.uint16).reshape(img.size[1], img.size[0])
	# disp = plt.imread('dataRGBD/Disparity'+str(dataset)+'/disparity'+str(dataset)+'_'+str(disp_idx+1)+'.png')
	img = Image.open('dataRGBD/RGB'+str(dataset)+'/rgb'+str(dataset)+'_'+str(rgb_idx+1)+'.png')
	rgb = np.array(img)
	disp_down = ndimage.uniform_filter(disp,size=5,mode='constant')
	rgb_down = np.zeros((rgb.shape))
	for dim in range(3): rgb_down[:,:,dim] = ndimage.uniform_filter(rgb[:,:,dim],size=5,mode='constant')
	for i in np.arange(1,640,5):
		for j in np.arange(1,480,5):
			z = np.array([i,j,1])
			d = disp_down[j,i]
			dd = -0.00304*d + 3.31
			depth = 1.03/dd
			if (depth<0 or depth>5):continue
			rgbi = np.int16(np.round((i*526.37+dd*(-4.5*1750.46)+ 19276.0)/585.051))
			rgbj = np.int16(np.round((j*526.37 + 16662.0)/585.051))
			if (rgbj<0 or rgbj>480) or (rgbi<0 or rgbj>640):continue
			R = np.matmul(Rwc,Roc.T)
			m = np.matmul(R,np.matmul(invK,z))*depth + pwc
			if m[2]< thres: 
				m_grid = np.array([np.int(np.round((m[0]-xmin)/res)), np.int(np.round((m[1]-ymin)/res))])
				color_grid[m_grid[0],m_grid[1],:] = rgb_down[rgbj,rgbi] + color_grid[m_grid[0],m_grid[1],:] # accumulated rgb
				color_grid_N[m_grid[0],m_grid[1]] = color_grid_N[m_grid[0],m_grid[1]] + 1 # count how many pixels mapped to this grid (average at the end)
	if (t%500==0):
		pixel_idx = np.where(color_grid_N!=0)
		color_grid_norm = np.zeros(color_grid.shape)
		for dim in range(3): 
			color_grid_norm[pixel_idx[0],pixel_idx[1],dim] = color_grid[pixel_idx[0],pixel_idx[1],dim]/color_grid_N[pixel_idx[0],pixel_idx[1]]
		fig,ax = plt.subplots(1,3,figsize=(15,5))
		plt.clf()
		plt.subplot(1,3,1)
		plt.imshow(lamda)
		plt.subplot(1,3,2)
		img = Image.fromarray(color_grid_norm.astype(np.uint8))
		plt.imshow(img)
		plt.subplot(1,3,3)
		plt.imshow(color_grid_N)
		plt.tight_layout()
		plt.savefig('Set'+str(dataset)+'t'+str(t)+'_NStation_Color_Map.png',dpi=500)
		plt.close()
	print(t)


# rgb_test = Image.fromarray(rgb_down.astype(np.uint8))
# plt.clf()
# plt.imshow(rgb_test)
# plt.savefig('rgb_test.png')

