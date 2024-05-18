# Real-Time-Visual-Servoing-Grasping
This project implements a Real-Time Visual Servoing framework that uses optical flow, DCEM sampling, and MPC to guide a robotic arm from an initial position to a target position for grasping tasks

### Installation and Setup

1. Make sure to pull submodules by running the following command:
   ```bash
   git submodule update --init --recursive

2. Create the conda environment (rtvs in this case) using the provided `environment.yml` file and activate it:
   
   ```bash
   conda env create -f environment.yml
   conda activate rtvs

3. Navigate to src/controllers/rtvs/flownet2-pytorch/networks/resample2d_package/ and run the following commands
   
   ```bash
   python3 setup.py build
   python3 setup.py install

4. To install flownet2 navigate to src/controllers/rtvs/flownet2-pytorch and run :
   ```bash
   bash install.sh
   
5. For 3D rendering , run the following command :
 
   ```bash
    pip install git+https://github.com/Improbable-AI/airobot@4c0fe31
   
### Demo

We demonstrate simulations for two cases: linear motion and circular motion of the belt.

#### 1) Linear Motion Case
In this case, the belt moves linearly.

![linearonline-video-cutter com-ezgif com-video-to-gif-converter](https://github.com/AniruthSuresh/Real-Time-Visual-Servoing-Grasping/assets/137063103/72a05d35-e348-4c68-a842-12ab26c73918)


#### 2) Circular Motion Case
Here, the belt moves in a circular motion.

![Circular_belt-ezgif com-video-to-gif-converter](https://github.com/AniruthSuresh/Real-Time-Visual-Servoing-Grasping/assets/137063103/630c88d0-c41a-4ec9-89b0-081d0249d9e0)

These simulations illustrate the behavior of the system under different belt motion scenarios. The linear motion case showcases how the system responds to straight-line movements, while the circular motion case demonstrates its behavior when dealing with rotational motion.


