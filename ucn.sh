#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ucn
export PYTHONPATH=$PYTHONPATH:/home/robotdev/franka_dev/UnseenObjectClustering/

cd /home/robotdev/franka_dev/
python ./UnseenObjectClustering/ros/ros_ucn.py --data_dir ./ros_data --camera_type eye_in_hand
