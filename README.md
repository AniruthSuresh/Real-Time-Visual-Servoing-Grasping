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
   
