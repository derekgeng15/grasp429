import cv2, numpy as np, tf.transformations as tft, tf
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
 
DICT_5x5_250 = 6
markerLen = 0.141
# ------------------------------
# ENTER YOUR PARAMETERS HERE:
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 4
SQUARE_LENGTH = 0.049
MARKER_LENGTH = 0.0245
LENGTH_PX = 640   # total length of the page in pixels
MARGIN_PX = 20    # size of the margin in pixels
SAVE_NAME = 'ChArUco_Marker.png'
# ------------------------------

# objPoints = np.array([(-markerLen/2, markerLen/2, 0), 
#              (markerLen/2, markerLen/2, 0), 
#              (markerLen/2, -markerLen/2, 0), 
#              (-markerLen/2, -markerLen/2, 0)])
def detect_pose(image, camera_matrix, dist_coeffs):
    # Undistort the image

    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Detect markers in the undistorted image
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)

    # If at least one marker is detected
    if len(marker_ids) > 0:
        print(len(marker_ids))
        # Interpolate CharUco corners
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)

        # If enough corners are found, estimate the pose
        if charuco_retval:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)
            return rvec, tvec

            # If pose estimation is successful, draw the axis
            # if retval:
            #     cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1, thickness=15)
    return None
 
# generates T for a given translation and quaternion rotation   
def gen_T(tvec, quat):
    T = tft.quaternion_matrix(quat)
    T[:3, 3] = tvec
    return T

# solves for T of marker in camera frame
def solveTFromImg(img, mtx):

    # arucoDict = cv2.aruco.getPredefinedDictionary(DICT_5x5_250)
    # arucoParam = cv2.aruco.DetectorParameters()

    # detector = cv2.aruco.ArucoDetector(arucoDict, arucoParam)

    # corners, ids, rejected = detector.detectMarkers(img)
    # # frame_markers = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    # # cv2.imshow(name, img)
    # _, rvec, tvec, = cv2.solvePnP(objPoints, corners[0], mtx, False, cv2.SOLVEPNP_ITERATIVE  )
    ret = detect_pose(img, mtx, np.array([0, 0, 0, 0, 0]))
    if ret is None:
        exit(1)
    rvec, tvec = ret[0], ret[1] 

    # corners = [(int(c[0]), int(c[1])) for c in corners[0].reshape((4, 2)).tolist()]
    # for i in range(4): 
    #     cv2.line(img, corners[i], corners[(i + 1)%4], (0, 255, 0), 2)
    # convert rvec to quaternion
    rotation_matrix = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1]],
                            dtype=float)
    rotation_matrix[:3, :3], _ = cv2.Rodrigues(rvec)
    rvec = tft.quaternion_from_matrix(rotation_matrix)
    return gen_T(tvec.reshape(3,), rvec)

bridge = CvBridge()
# inHandImg = cv2.imread("eye_in_hand_marker.jpg")
# toHandImg = cv2.imread("eye_to_hand_marker.jpg")
inHandImg = None
toHandImg = None

inHandCamMtx = np.array([922.0578002929688, 0.0, 645.614990234375, 0.0, 921.6668701171875, 356.9132385253906, 0.0, 0.0, 1.0]).reshape((3, 3))
toHandCamMtx = np.array([900.0070190429688, 0.0, 650.471435546875, 0.0, 899.6478271484375, 357.7978515625, 0.0, 0.0, 1.0]).reshape((3, 3))

def inHand_callback(msg):
    global inHandImg
    try:
        inHandImg = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        
def toHand_callback(msg):
    global toHandImg
    try:
        toHandImg = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))  
            
def writeLaunch(name, trans, quat):
    f = open(name, "w")
    f.write(f'''<launch>
  <node pkg="tf2_ros" type="static_transform_publisher" name="eye_to_hand_calibration"
      args="{trans[0]} {trans[1]} {trans[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]} panda_link0 eye_to_hand_link" />
</launch>''')
    f.close()
    
if __name__ == "__main__":
    rospy.init_node('cross_calibration')
    listener = tf.TransformListener()
    
    # get imgs from live camera topics 
    inHandSub = rospy.Subscriber("/eye_in_hand/color/image_raw", Image, inHand_callback)
    toHandSub = rospy.Subscriber("/eye_to_hand/color/image_raw", Image, toHand_callback)
    while inHandImg is None or toHandImg is None:
        continue
    inHandSub.unregister()
    toHandSub.unregister()
    
    # calculate intermediate transforms
    inHandOptical_target = solveTFromImg(inHandImg, inHandCamMtx)
    target_toHandOptical = tft.inverse_matrix(solveTFromImg(toHandImg, toHandCamMtx))
    link0_inHandOptical = gen_T(*listener.lookupTransform('/panda_link0', '/eye_in_hand_color_optical_frame', rospy.Time(0)))
    toHandOptical_toHandLink = gen_T(*listener.lookupTransform('/eye_to_hand_color_optical_frame', '/eye_to_hand_link', rospy.Time(0)))
    
    # link0_inHandOptical = gen_T(np.array((0.451, 0.032, 0.600)).reshape(3, ), np.array((-0.683, 0.730, 0.010, -0.031)))
    # toHandOptical_toHandLink = gen_T(np.array((-0.000, 0.014, -0.005)).reshape(3, ), np.array((0.488, -0.497, 0.508, 0.507)))

    # compute link0 -> eye_to_hand_link transform and write to file
    link0_toHandLink =  link0_inHandOptical @ inHandOptical_target @ target_toHandOptical @ toHandOptical_toHandLink
    writeLaunch("eye_to_hand_calibration.launch", link0_toHandLink[:3, 3], tft.quaternion_from_matrix(link0_toHandLink))
