# Real-Time-Visual-Servoing-Grasping
This project implements a Real-Time Visual Servoing framework that uses optical flow, DCEM sampling, and MPC to guide a robotic arm from an initial position to a target position for grasping tasks

### Installation and Setup

1. Make sure to pull submodules after cloning it by running the following command:
   ```bash
   git submodule update --init --recursive

2. Create the conda environment (rtvs in this case) using the provided `environment.yml` file and activate it:
   
   ```bash
   conda env create -f environment.yml
   conda activate rtvs

3. Navigate to ```src/controllers/rtvs/flownet2-pytorch/networks/resample2d_package/``` and run the following commands
   
   ```bash
   python3 setup.py build
   python3 setup.py install

4. To install flownet2 navigate to src/controllers/rtvs/flownet2-pytorch and run :
   ```bash
   bash install.sh
   
5. For 3D rendering , run the following command :
 
   ```bash
    pip install git+https://github.com/Improbable-AI/airobot@4c0fe31

6. To run the simulation :
   ```bash
   cd src
   python3 grasp.py --gui

#### Explanation of the Project

![image](https://github.com/AniruthSuresh/Real-Time-Visual-Servoing-Grasping/assets/137063103/f09d22dd-c5a4-41d8-adf4-135916f2d70a)


Our project involves real-time visual servoing and grasping tasks. We perform visual servoing and grasping by capturing the current image and comparing it with the destination image using optical flow techniques.

#### Visual Servoing:
Visual servoing is a technique used in robotics to control the movement of a robot based on visual feedback. In our project, we utilize optical flow to track the motion of objects in the environment. By comparing the current image with the desired destination image, we calculate the necessary motion commands to achieve the desired task, such as reaching a target position or grasping an object.

#### Grasping:
Grasping is the process of closing the robot's gripper around an object. We employ a combination of visual information and tactile feedback to perform grasping tasks accurately and efficiently.

#### DCEM (Differential Cross-Entropy Method):
DCEM is a stochastic optimization algorithm commonly used for reinforcement learning and optimization problems. In our project, we apply the DCEM algorithm to obatin predicted optical flows. By training a neural network using the differential cross-entropy method, we obtain a refined prediction that enhances the accuracy of our visual servoing and grasping tasks. In every iteration the mean and the gaussian is updated so as to get a better result teh next iteration .

#### Controller Velocity Adjustment:
The predicted flow loss obtained from the neural network is compared with the actual flow loss. Based on this comparison, the velocity commands sent to the controller are adjusted to minimize the discrepancy between the predicted and actual values. This ensures precise and robust control of the robot's movements during visual servoing and grasping operations.


### Demo

We demonstrate simulations for two cases: linear motion and circular motion of the belt.

#### 1) Linear Motion Case
In this case, the belt moves linearly.

![linearonline-video-cutter com-ezgif com-video-to-gif-converter](https://github.com/AniruthSuresh/Real-Time-Visual-Servoing-Grasping/assets/137063103/72a05d35-e348-4c68-a842-12ab26c73918)


#### 2) Circular Motion Case
Here, the belt moves in a circular motion.

![Circular_belt-ezgif com-video-to-gif-converter](https://github.com/AniruthSuresh/Real-Time-Visual-Servoing-Grasping/assets/137063103/630c88d0-c41a-4ec9-89b0-081d0249d9e0)


These simulations illustrate the behavior of the system under different belt motion scenarios. The linear motion case showcases how the system responds to straight-line movements, while the circular motion case demonstrates its behavior when dealing with rotational motion.

### Note: Model Availability

In this project repository, the `flownet2` model is not readily available due to its large size. Therefore, it's crucial to generate the `flownet2` model separately and add it to the data directory before proceeding with the project setup.


