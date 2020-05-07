# openpose-drone-control
Drone control scheme using OpenPose and neural net classification. 

## Setup

1. Install OpenPose.
2. Clone this repository to the same folder that contains OpenPose.
3. Install ROS, MAVROS, Gazebo, and ArduCopter. Make sure to source `/opt/ros/melodic/setup.bash` and export `ardupilot/Tools/autotest` in order to use ROS and ArduCopter commands.

## Launch

To launch, run the following commands in separate terminals:

```
sim_vehicle.py -v ArduCopter -f gazebo-iris --map --console
gazebo --verbose worlds/iris_arducopter_runway.world
roslaunch mavros apm.launch
python3 pose.py
```

To retrain the classification model, do the following:

1. Open `build_dataset.py` and edit the line `pose = ...` to the pose name that you want to build a new dataset for. (note: valid pose names include `['forward', 'back', 'turnleft', 'turnright', 'up', 'down', 'left', 'right', 'neutral']`)
2. Run `python3 build_dataset.py`, then hold that pose until the script ends. 
3. Run `python3 train_model.py`.

## Extra

For the camera to follow the drone, replace the camera section of `/usr/share/gazebo-9/worlds/iris_arducopter_runway.world` (or wherever your world is located) with the following:

```<camera name="user_camera">
      <camera name="user_camera">
        <track_visual>
          <name>iris_demo</name>
          <static>true</static>
          <use_model_frame>true</use_model_frame>
          <xyz>-3 0 0</xyz>
          <inherit_yaw>true</inherit_yaw>
        </track_visual>
      </camera>
```