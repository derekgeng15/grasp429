import numpy as np, tf.transformations as tft, tf
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

bridge = CvBridge()

rgbImg = None
rgbMsg = None

depthImg = None
depthMsg = None
def rgb_callback(msg):
    global rgbImg, rgbMsg
    try:
        rgbMsg = msg
        rgbImg = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))  
def depth_callback(msg):
    global depthImg, depthMsg
    try:
        depthMsg = msg
        depthImg = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e)) 
if __name__ == "__main__":
    camera = 'eye_to_hand'
    rospy.init_node('getdata')
    listener = tf.TransformListener()
    
    # get imgs from live camera topics 
    rgbSub = rospy.Subscriber(f"/{camera}/color/image_raw", Image, rgb_callback)
    depthSub = rospy.Subscriber(f"/{camera}/aligned_depth_to_color/image_raw", Image, depth_callback)
    
    while rgbImg is None or rgbMsg is None or depthImg is None or  depthMsg is None:
        continue
    rgbSub.unregister()
    rgbdOut = {
        'rgb': rgbImg,
        'depth' : depthImg,
        'msg_rgb': rgbMsg,
        'msg_depth' : depthMsg,
        'frame_stamp' : rgbMsg.header.stamp
    }
    np.save('./ros_data/rgbd', rgbdOut, allow_pickle=True)
    # rgbd = np.load('./franka_6d_grasp/data_bk/rgbd.npy', allow_pickle=True).tolist()

    # print(rgbd.keys())
    # print(type(rgbd['msg_rgb']))
    # print(rgbd['rgb'].shape)
    # print(type(rgbd['frame_stamp']))
    # print(rgbMsg.header.stamp)