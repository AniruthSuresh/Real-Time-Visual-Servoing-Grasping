# Real-Time-Visual-Servoing-Grasping
This project implements a Real-Time Visual Servoing framework that uses optical flow, DCEM sampling, and MPC to guide a robotic arm from an initial position to a target position for grasping tasks

### Installation and Setup

1. Create the conda environment (rtvs in this case) using the provided `environment.yml` file and activate it:
   
   ```bash
   conda env create -f environment.yml
   conda activate rtvs

2. Navigate to src/controllers/rtvs/flownet2-pytorch/networks/resample2d_package/ and run the following commands
   
   ```bash
   python3 setup.py build
   python3 setup.py install

3. To install flownet2 navigate to src/controllers/rtvs/flownet2-pytorch and run :
   ```bash
   bash install.sh
   

   
