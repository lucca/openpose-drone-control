# openpose-drone-control
Drone control scheme using OpenPose and neural net classification. 

## Setup

1. Install OpenPose.
2. Clone this repository to the same folder that contains OpenPose.
3. Install ROS, MAVROS, Gazebo, and ArduCopter. Make sure to source `/opt/ros/melodic/setup.bash` and export `ardupilot/Tools/autotest` in order to use ROS and ArduCopter commands.

## Launch

To launch, run the following commands in separate terminals:

```bash
sim_vehicle.py -v ArduCopter -f gazebo-iris --map --console
gazebo --verbose obstacle.world
roslaunch mavros apm.launch
python3 pose.py
```

To retrain the classification model, do the following:

1. Open `build_dataset.py` and edit the line `pose = ...` to the pose name that you want to build a new dataset for. (note: valid pose names include `['forward', 'back', 'turnleft', 'turnright', 'up', 'down', 'left', 'right', 'neutral']`)
2. Run `python3 build_dataset.py`, then hold that pose until the script ends. 
3. Run `python3 train_model.py`.
