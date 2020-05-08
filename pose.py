# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import rospy 
from mavros_msgs.srv import SetMode, CommandBool, CommandTOL
from geometry_msgs.msg import TwistStamped, Vector3


# ================ ROS Setup ========================
rospy.init_node('mavros_final_project')
rate = rospy.Rate(10)

commandVelocityPub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size = 10)
setVelocity = TwistStamped()

def setMode():
    print("Setting Mode to Guided: mode guided")
    rospy.wait_for_service('/mavros/set_mode')
    try: 
        mavSetMode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        mavSetMode(custom_mode="Guided")
    except rospy.ServiceException as e:
        print(e)

def armCopter():
    print("Arming throttle: arm throttle")
    rospy.wait_for_service('/mavros/cmd/arming')
    try:
        mavArm = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        mavArm(value = True)
    except rospy.ServiceException as e:
        print(e)

def takeOff():
    print("Taking off: takeoff 1")
    try:
        mavTakeOff = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
        mavTakeOff(altitude=1)
    except rospy.ServiceException as e:
        print(e)

def land():
    print("Landing: mode land")
    try:
        mavLand = rospy.ServiceProxy('/mavros/cmd/land', CommandTOL)
        mavLand(altitude=1)
    except rospy.ServiceException as e:
        print(e)

def disarmCopter():
    print("Disarming throttle: disarm")
    try:
        mavDisarm = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        mavDisarm(value = False)
    except rospy.ServiceException as e:
        print(e)


# ================ Classification setup ========================
def right():
    setVelocity.twist.linear.y = -1

def left():
    setVelocity.twist.linear.y = 1

def forward():
    setVelocity.twist.linear.x = 1

def back():
    setVelocity.twist.linear.x = -1

def down():
    setVelocity.twist.linear.z = -0.5

def up():
    setVelocity.twist.linear.z = 0.5

def turnleft():
    setVelocity.twist.angular.z = 0.5

def turnright():
    setVelocity.twist.angular.z = -0.5

def neutral():
    pass

ops = {
    "right": right,
    "left": left,
    "forward": forward,
    "back": back,
    "down": down,
    "up": up,
    "turnleft": turnleft,
    "turnright": turnright,
    "neutral": neutral
}


# ================ OpenPose ========================
try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../openpose/build/python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../openpose/examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../openpose/models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Import classification model
    mlp = load('model.joblib')

    # Set up camera processing
    datum = op.Datum()
    stream = cv2.VideoCapture(0)

    frame_rate = 3
    prev = 0

    setMode()
    armCopter()
    takeOff()
    time.sleep(5)

    while(True):
        # Limit framerate
        time_elapsed = time.time() - prev
        res, image = stream.read()

        if time_elapsed > 1./frame_rate:
            prev = time.time()

            setVelocity.twist.linear = Vector3(x = 0, y = 0, z = 0)
            setVelocity.twist.angular = Vector3(x = 0, y = 0, z = 0)

            datum.cvInputData = image
            opWrapper.emplaceAndPop([datum])

            pose = ""

            # Extract keypoints
            kp = datum.poseKeypoints
            kp_overlay = datum.cvOutputData
            if kp.shape: # If 0 people in image, shape doesn't exist
                kp = kp[0][1:9] # Extract upper body indices
                kp = np.delete(kp, 2, 1) # Drop confidence score
                kp = kp.flatten()
                kp = kp.reshape(1, -1)
            
                pose = mlp.predict(kp)[0]
            else:
                pose = "neutral"

            ops[pose]()
            commandVelocityPub.publish(setVelocity)

            cv2.imshow("OpenPose 1.6.0 - Chad API", kp_overlay)
            if cv2.waitKey(1) == ord('q'):
                stream.release()
                cv2.destroyAllWindows()
                break
    
    land()
    disarmCopter()
    time.sleep(5)

except Exception as e:
    print(e)
    sys.exit(-1)
