This is course project of [ECE276A(UCSD)](https://natanaso.github.io/ece276a/). Thanks for the instructor and TAs for providing the test dataset and help.
### [Formal report](./SLAM_KL.pdf)
* Problem formulation
* Algorithm description
* Test results on three datasets

### Sensors on the robot.
* [Robot configuration](./RobotConfiguration.pdf)
* Encoder(40Hz): distance travelled by the wheelp
* IMU(100Hz): linear/angular velocity
* Hokuyo LIDAR(40Hz): (double) distance from enviroment obstacles
* Camera(20Hz): video of the surrounding environment, only for texture mapping in this project. Data not uploaded due to file size limit.

### Goal of SLAM
* Determine the self-location of th robot
* Represent the environment (e.g. walls/obstacles) 
* Mapping and localization will be performed at the same time because they are mutually dependen


### Main scripts
* Proj2.py: For prediction and update
	* Change accordingly in the script: 
		* dataset: 
		* N: particle number (when dataset=20 and N=10)
	* Reads in: 
		* Encoders20.npz
		* Hokuyo20.npz
		* Imu20.npz
	* Saves: 
		* Set20_N10_t*.png:
			* saved every 250 steps
			* plot accumulated log-odds ratio map, moving traces in the grid and particle position in physical location
		* Set20_N10_lamda.npy:
			Final log-odds ratio map
		* Set20_N10_CurrentBest.npy
			The weighted phycial position at every time step

* TextureM.py: for texture mapping
	* Change accordingly in the script: 
		* dataset: 
		* N: particle number (when dataset=20 and N=10)
	* Reads in: 
		* Encoders20.npz
		* Kinect20.npz
		* dataRGBD/Disparity/disparity20_*.png
		* dataRGBD/RGB/rgb20_*.png
	* Saves:
		* Set20_t*_Color_Map.png:
			* saved every 500 steps
			* plot accumulated log-odds ratio map, mapped pixels in the grid and Number of pixels corresponding to each grid
