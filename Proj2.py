import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot  as plt
import matplotlib.cm as cm
import os 
from map_utils import bresenham2D, mapCorrelation
import random

def m_bar(mu,current_lidar_ranges):
  '''
  Input: 
    mu: current particle position
    current_lidar_ranges: current lidar readings
  Output: 
    m_bar0: physical location of obstacles in world frame
  '''
  m_bar0 = np.zeros((1081,2)) # obstacle location in robot frame
  alpha = np.arange(lidar_angle_min,lidar_angle_max+0.0001,lidar_angle_increment) + mu[2] 
  m_bar0[:,0] = current_lidar_ranges * np.cos(alpha) + mu[0]
  m_bar0[:,1] = current_lidar_ranges * np.sin(alpha) + mu[1]
  return m_bar0

# Keep a grid map of 1100*1100, 
# cover -55 meter to 55 meter in both x and y axis
# center is at (550,550)
def lamda_delta(mu,current_lidar_ranges):
  '''
    Input: 
      current_lidar_ranges: current lidar ranges from lidar frame
      mu: robot state: x,y,theta
    Output: 
      lamda_delta: odds ratio update from current scan
      end: obstacle positive in physical position
      end_grid: obstacle position in the grid map 
      start: robot position in the grid
  '''
  lamda_delta = np.zeros((1100,1100)) # log(g_h)
  xmin = -55 # meter
  xmax = 55
  ymin = -55
  ymax = 55
  m_res = 0.1 # meters/grid
  alpha = np.arange(lidar_angle_min,lidar_angle_max+0.0001,lidar_angle_increment) + mu[2]
  end = np.zeros((1081,2)) # physical location of obstacle in world frame
  end[:,0] = current_lidar_ranges*np.cos(alpha) + mu[0]
  end[:,1] = current_lidar_ranges*np.sin(alpha) + mu[1]
  end_grid = np.zeros((1081,2)) # obstacle in grid location
  end_grid[:,0] = np.rint((end[:,0]-xmin)/m_res)
  end_grid[:,1] = np.rint((end[:,1]-ymin)/m_res)
  end_grid = end_grid.astype(int)
  start = np.int16(np.round((mu[:2] - np.array([xmin,ymin]))/m_res)) # robot position in grid location
  for i in range(end_grid.shape[0]):
    trace = bresenham2D(start[0],start[1],end_grid[i,0],end_grid[i,1])
    trace = trace.astype(int)
    if trace.shape[1] < 2: continue # remove very close observations
    lamda_delta[trace[0,:-1],trace[1,:-1]] = np.log(1/4)
    lamda_delta[trace[0,-1],trace[1,-1]] = np.log(4)
  lamda_delta[lamda_delta<np.log(1/4)] = np.log(1/4)
  return lamda_delta, end, end_grid, start

def motion(s0,i):
  '''
  Input: 
    s0: previous state variable: x(meter),y(meter),theta(rad)
    i: ith measurement from encoder 
  Output: 
    s_tp1: updated state variable
  '''
  time_1 = encoder_stamps[i] # Encoder at t
  time_2 = encoder_stamps[i+1] # Encoder at t+1
  dt = time_2 - time_1
  imu_current = imu_angular_velocity[2,np.all([imu_stamps<time_2,imu_stamps>time_1],axis=0)]
  w = np.mean(imu_current) # mean w from encoder(t) to encoder(t+1)
  if imu_current.shape[0]==0: s_tp1 = s0
  else: 
    v = np.mean(encoder_counts[:,i],axis=0)*0.0022/dt # meters/second
    v = v+np.random.normal(0,0.1*abs(v)) # (default 0.1) (0.2+0.1abs(w) BBN) (0.15+0.05, BMN)
    w = w+np.random.normal(0,0.03*abs(w)) #(0.05 MN, 0.03, 0.1 BN )
    # stationary noise
    # v = v + np.random.normal(0,0.4)
    # w = w + np.random.normal(0,0.03) #(0.03 for set 20 and 21)
    s_tp1 = np.zeros((3))
    s_tp1[0] = s0[0]+v/w*(np.sin(s0[2]+w*dt)-np.sin(s0[2]))
    s_tp1[1] = s0[1]+v/w*(np.cos(s0[2])-np.cos(s0[2]+w*dt))
    s_tp1[2] = s0[2]+w*dt
  return s_tp1


def SIR(mu_tp1,alpha2_tp1):
  '''
  Input:
    mu_tp1: after update particles value, x,y,theta in world frame
    alpha2_tp1: after update weights
  Output: 
    mu_re: resampled mu
    alpha_re: resampled weights
  '''
  # print('Resampling...at t='+str(t))
  mu_re = np.zeros(mu_tp1.shape)
  j=0;c=alpha2_tp1[0];N=mu_tp1.shape[1]
  for k in range(N):
    v = np.random.rand(1)/float(N);beta = v+k/float(N)
    while beta > c: j+=1; c+=alpha2_tp1[j]
    mu_re[:,k] = mu_tp1[:,j]
  alpha2_re = 1/float(N)*np.ones((N));
  return mu_re, alpha2_re

# Test for SIR
# N=50
# mu_tp1 = np.random.rand(2,N); alpha2_tp1 = np.exp(np.arange(N)); alpha2_tp1 = alpha2_tp1/np.sum(alpha2_tp1)
# [mu_re,alpha2_re] = SIR(mu_tp1,alpha2_tp1)
# fig,ax=subplots(1,2,figsize=(10,5))
# plt.subplot(1,2,1)
# plt.scatter(mu_tp1[0,:],mu_tp1[1,:],s=alpha2_tp1*50)
# plt.subplot(1,2,2)
# plt.scatter(mu_re[0,:],mu_re[1,:],s=alpha2_re*50)
# plt.xlim(0,1);plt.ylim(0,1)
# plt.savefig('SIR_test.png',dpi=250)



dataset = 20
with np.load("Encoders%d.npz"%dataset) as data:
  encoder_counts = data["counts"] # 4 x n encoder counts
  encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load("Hokuyo%d.npz"%dataset) as data:
  lidar_angle_min = data["angle_min"] # start angle of the scan [rad], -135 degree in rad
  lidar_angle_max = data["angle_max"] # end angle of the scan [rad], 135 degree in rad
  lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad], 270/1080 degree in rad
  lidar_range_min = data["range_min"] # minimum range value [m]
  lidar_range_max = data["range_max"] # maximum range value [m]
  lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
  lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans

with np.load("Imu%d.npz"%dataset) as data:
  imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
  imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
  imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

# with np.load("Kinect%d.npz"%dataset) as data:
#   disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
#   rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images


# fig,ax=plt.subplots(4,1)
# plt.subplot(4,1,1)
# plt.plot(encoder_stamps,encoder_counts[0,:])
# plt.title('Encoder FR')
# plt.subplot(4,1,2)
# plt.plot(encoder_stamps,encoder_counts[1,:])
# plt.title('FL')
# plt.subplot(4,1,3)
# plt.plot(encoder_stamps,encoder_counts[2,:])
# plt.title('RR')
# plt.subplot(4,1,4)
# plt.plot(encoder_stamps,encoder_counts[3,:])
# plt.title('RL')
# plt.tight_layout()
# # plt.show()
# plt.savefig('EncoderReading.png')
# print('FR traveled '+str(np.sum(encoder_counts[0,:])*0.0022)+' meters')
# print('FL traveled '+str(np.sum(encoder_counts[1,:])*0.0022)+' meters')
# print('RR traveled '+str(np.sum(encoder_counts[2,:])*0.0022)+' meters')
# print('RL traveled '+str(np.sum(encoder_counts[3,:])*0.0022)+' meters')
# print('Right wheels traveled '+str((np.sum(encoder_counts[0,:])+np.sum(encoder_counts[2,:]))/2*0.0022)+' meters')
# print('Left wheels traveled '+str((np.sum(encoder_counts[1,:])+np.sum(encoder_counts[3,:]))/2*0.0022)+' meters')
# # Right wheels travel more -> More left turns -> left turn is positive

# fig,ax=plt.subplots(3,1)
# plt.subplot(3,1,1)
# plt.plot(imu_stamps,imu_angular_velocity[0,:])
# plt.title('IMU angular velocity 1')
# plt.subplot(3,1,2)
# plt.plot(imu_stamps,imu_angular_velocity[1,:])
# plt.title('IMU angular velocity 2')
# plt.subplot(3,1,3)
# plt.plot(np.arange(imu_angular_velocity.shape[1]),imu_angular_velocity[2,:])
# plt.title('IMU angular velocity 3 (yaw rate)')
# plt.tight_layout()
# # plt.show()
# plt.savefig(str(dataset)+'IMU_Angular_Velocity.png')


### First Mapping

## Sensors in robot frame 
##  relative to robot center, at the lidar location
##  x axis, to front; y axis, to left; z axis, to the sky; theta, left is positive
# IMU, assume the same as robot frame, 12187 time points
# Encoder, 40Hz, 4956 time points
# p_RL = np.array([(311.15+(476.25-311.15)/4),0.,0.])*10e-3
# p_RR = -p_RL
# p_FL = p_RL + np.array([0.,330.20,0.])*10e-3
# p_FR = p_RR + np.array([0.,330.20,0.])*10e-3
# Lidar,  4962 time points, 1081 beams
# orientation is the same as the robot
# p_Li = np.array([0.,298.33,514.35])*10e-3
# RGBD camera, This is relative to rear wheel center, change to relative to lidar
# p_rgbd = np.array([0.18,0.005,0.36]) + 
# R_rgbd = np.array([0.,0.36,0.021]) + # roll, pitch, yaw in rad



## Robot pose relative to world frame
# At t=0, robot frame is the same as world frame
# R_r = np.diag(np.array([1.,1.,1.]))
# p = np.array([0.,0.,0.])
# At t=0, map from the first lidar scan and initialize the map, no noise
# Map in world frame


# Map in grids, check the odds raio map at t=0
# [lamda,end,end_grid,start] = lamda_delta(np.array([0.,0.,0.]),lidar_ranges[:,0]) # accumulation odds ratio
# gamma_grid = 1-1/(1+np.exp(lamda))
# m_grid = np.random.binomial(1,gamma_grid,gamma_grid.shape)
# fig,ax = plt.subplots(2,2)
# plt.subplot(2,2,1)
# plt.scatter(end[:,0],end[:,1],s=0.5)
# plt.subplot(2,2,2)
# plt.scatter(end_grid[:,0],end_grid[:,1],s=0.5)
# plt.subplot(2,2,3)
# plt.imshow(lamda[min(end_grid[:,0]):max(end_grid[:,0]),min(end_grid[:,1]):max(end_grid[:,1])].T)
# plt.subplot(2,2,4)
# plt.imshow(m_grid[min(end_grid[:,0]):max(end_grid[:,0]),min(end_grid[:,1]):max(end_grid[:,1])].T)
# plt.show()


### Prediction-only particle filter
# Try one particle
# use encoder time stamp 
# s = np.zeros((encoder_stamps.shape[0],3)) # x(meter),y(meter),theta(rad)
# for i in range(encoder_stamps.shape[0]-1):
#   s0 = s[i,:]
#   s_tp1 = motion(s0,i)
#   s[i+1] = s_tp1
#   print(i,s[i+1])

# colors = cm.rainbow(np.linspace(0,1,s.shape[0]))
# fig, ax = plt.subplots()
# ax.scatter(s[:,0],s[:,1],s=5,color=colors)
# plt.title('Single particle prediction results')
# plt.savefig('Set'+str(dataset)+'Single_Trace.png')
# # plt.savefig('Set'+str(dataset)+'Single_Trace_MN.png')
# plt.close()

### Update step with prediction


# Settings and initialization
N = 10 # Particle number
xim = np.arange(-55,55+0.1,0.1) # physical boundary of the map grid
yim = np.arange(-55,55+0.1,0.1)
Mu = [] # x,y,theta (physcial location in world frame)
Alpha2 = [] # weight
Lamda_Map = [] # Accumulated odds ratio
BestMu = np.zeros((1100,1100)) # The traveling trace of the robot in a grid
CurrentBest = np.zeros((3,encoder_stamps.shape[0]-1)) # weighted mu for every step
# Initialization of t=0
mu = np.zeros((3,N))
alpha2 = 1/N*np.ones((N))
[lamda,end,end_grid,start] = lamda_delta(mu[:,0],lidar_ranges[:,0]) # accumulation odds ratio
gamma_grid = 1-1/(1+np.exp(lamda))
m_grid = np.random.binomial(1,gamma_grid,gamma_grid.shape)
Mu.append(mu)
Alpha2.append(alpha2)
Lamda_Map.append(lamda)

# fig,ax=plt.subplots(1,1)
# plt.imshow(lamda)
# plt.show()


for t in range(encoder_stamps.shape[0]-1):
  current_time = encoder_stamps[t]  
  lidar_idx = np.argmin(np.abs(lidar_stamsp-current_time)) # the closest time point in lidar stamp
  # if encoder_counts[0,t] == 0: continue # don't predict and update if no velocity
  # Initialize t+1 observations
  mu_tp1 = np.zeros((3,N))
  alpha2_tp1 = np.zeros((N))
  Samesum = np.zeros((N))
  for i in range(N):
  # First move
    mu_tp1[:,i] = motion(mu[:,i],t)
    # Observe and update
    m_bar0 = m_bar(mu_tp1[:,i],lidar_ranges[:,lidar_idx]) 
    vp = m_bar0.T # physical obstacle position in world frame
    Nvib = 5
    xs = np.random.rand(Nvib)*0.001-0.0005 # robot position x-vibrations, in physical location
    ys = np.random.rand(Nvib)*0.001-0.0005 # robot position y-vibrations
    # 0.001 SSV
    Corr = mapCorrelation(m_grid,xim,yim,vp,xs,ys)
    Corr_idx = np.argmax(Corr); idx_x = Corr_idx % Nvib; idx_y = np.remainder(Corr_idx,Nvib)
    mu_tp1[0,i] = mu_tp1[0,i] + xs[idx_x]
    mu_tp1[1,i] = mu_tp1[1,i] + ys[idx_y]
    Samesum[i] = np.max(Corr) # choose the max as the correlation
  # print(Samesum)
  # print(i)
  Samesum = Samesum-np.max(Samesum)
  alpha2_tp1 = np.multiply(alpha2,np.exp(Samesum))
  alpha2_tp1 = alpha2_tp1 / np.sum(alpha2_tp1)
  # Resample after update
  if 1/np.sum(alpha2_tp1**2) < N/2:
    print('Resampling...at t='+str(t))
    [mu_re,alpha2_re] = SIR(mu_tp1,alpha2_tp1)
    mu_tp1 = mu_re; alpha2_tp1 = alpha2_re
  # Update the map from the most likely particle
  CurrentBest[:,t] = np.sum(np.multiply(mu_tp1.T, alpha2_tp1.T[:,None]),axis=0)
  print(CurrentBest[:,t])
  [lamda_d, end, end_grid, start]=lamda_delta(CurrentBest[:,t],lidar_ranges[:,lidar_idx])
  BestMu[start[0],start[1]] = 1
  lamda = lamda + lamda_d
  gamma_grid = 1-1/(1+np.exp(lamda))
  m_grid = np.random.binomial(1,gamma_grid,gamma_grid.shape)
  if (t%500==0):
    print(t);
    Mu.append(mu_tp1);Alpha2.append(alpha2_tp1);Lamda_Map.append(lamda)
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(lamda)
    plt.subplot(1,3,2)
    plt.imshow(BestMu)
    plt.subplot(1,3,3)
    plt.scatter(mu_tp1[0,:],mu_tp1[1,:],s=alpha2_tp1*50)
    # plt.scatter(start[1],1100-start[0],s=2,marker='x', color='red')
    plt.savefig('Set'+str(dataset)+'_N'+str(N)+'_t'+str(t)+'_NStation_SSVTest.png',dpi=500)
    plt.close()
  mu = mu_tp1
  alpha2 = alpha2_tp1

np.save('Set'+str(dataset)+'_N'+str(N)+'_NStation_SSVTest_CurrentBest.npy', CurrentBest)
np.save('Set'+str(dataset)+'_N'+str(N)+'_NStation_SSVTest_lamda.npy', lamda)

# BBN_SV: big v noise, big w noise, small vibration
# BBN_SSV: big v noise, big w noise, small small vibration
# Test: add vibration movement

# Plot the omega trace
# fig,ax = plt.subplots(1,1)
# plt.plot(BestOmega)
# plt.xlabel('Encoder #')
# plt.savefig('Set'+str(dataset)+'_N'+str(N)+'_Bestomega.png')

