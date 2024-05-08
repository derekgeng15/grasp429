#!/user/bin/env python3

import numpy as np
import ros_numpy as rnp
import os
import sys
import rospy
import time
import message_filters
import moveit_commander
import geometry_msgs.msg
import franka_gripper.msg
import actionlib
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger, TriggerRequest
from moveit_msgs.msg import PlanningScene, Grasp
from moveit_msgs.srv import GetPlanningSceneRequest, GetPlanningScene
from menpo.shape import TriMesh

import trimesh

import tf.transformations as tr, tf
from visualization_utils import visualize_grasps

import threading
lock = threading.Lock()


def from_matrix_to_pose(mtx):
    translation = tr.translation_from_matrix(mtx)
    quaternion = tr.quaternion_from_matrix(mtx)

    pose = geometry_msgs.msg.Pose()
    pose.orientation.w = quaternion[3]
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.position.x = translation[0]
    pose.position.y = translation[1]
    pose.position.z = translation[2]
    return pose


def from_pose_to_matrix(pose):
    T = tr.quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T
def gen_T(tvec, quat):
    T = tr.quaternion_matrix(quat)
    T[:3, 3] = tvec
    return T


class Franka6DGraspCommand(object):
    def __init__(self, data_dir, camera_type, debug=False):
        self._debug = debug

        # moveit_commander robot_arm
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('franka_6d_grasp', anonymous=True)
        self.robot_arm_commander = moveit_commander.RobotCommander()
        self.planning_scene_interface = moveit_commander.PlanningSceneInterface()
        self.group_name = "panda_arm"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self._add_setup_scene()

        # Demo of planning_scene collision_matrix, might be useful for the demo.
        planning_scene_service = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
        req = GetPlanningSceneRequest()
        planning_scene = planning_scene_service(req)
        print(planning_scene.scene.allowed_collision_matrix)

        if not self._debug:
            # gripper_commander robot_gripper
            self.robot_gripper_commander = actionlib.SimpleActionClient('/franka_gripper/grasp',
                                                                        franka_gripper.msg.GraspAction)
            self.robot_gripper_commander.wait_for_server()

        # camera subscriber
        rgb_sub = message_filters.Subscriber(f'/{camera_type}/color/image_raw', Image, queue_size=10)
        depth_sub = message_filters.Subscriber(f'/{camera_type}/aligned_depth_to_color/image_raw', Image, queue_size=10)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 1, 0.1)
        ts.registerCallback(self.callback_rgbd)
        self.data_dir = data_dir

        # save camera_info
        self.camera_info = {}
        self.network_input = {}

        msg = rospy.wait_for_message(f'/{camera_type}/color/camera_info', CameraInfo)
        self.camera_info['width'] = msg.width
        self.camera_info['height'] = msg.height
        self.camera_info['K'] = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
        print(self.camera_info['K'])
        os.makedirs(self.data_dir, exist_ok=True)
        np.save(f'{self.data_dir}/camera_info.npy', self.camera_info)

        # grasp inference services
        rospy.wait_for_service('/ucn_inference')
        self.seg_service = rospy.ServiceProxy('/ucn_inference', Trigger)
        rospy.wait_for_service('/contact_grasp_inference')
        self.grasp_service = rospy.ServiceProxy('/contact_grasp_inference', Trigger)

        # save tf info of T_{cam_optical}{end_effector_link}
        # Todo: Get tf by subscribing ros messages instead of hardcoded value.
        # Todo: Remember to set the corresponding ee_link based on the calibration TF.
        # self.tf_tcp_to_optical = tr.translation_matrix([0.0417, -0.040, -0.0430]) @ \
        # tr.quaternion_matrix([0.013, 0.0143, 0.701, 0.712855])
        listener = tf.TransformListener()
        # print(listener.allFramesAsString())
        listener.waitForTransform('/panda_link0', f'/{camera_type}_color_optical_frame',  rospy.Time(0), rospy.Duration(4.0))
        # tvec, quat = 
        self.tf_optical_to_base = gen_T(*listener.lookupTransform('/panda_link0',f'/{camera_type}_color_optical_frame',  rospy.Time(0)))
        self.move_group.set_end_effector_link("panda_hand_tcp")

        # set motion planning parameters
        # Todo: Adjust these parameters.
        self.move_group.set_goal_position_tolerance(0.005)
        self.move_group.set_goal_orientation_tolerance(0.02)
        self.move_group.set_planning_time(2)
        self.move_group.set_pose_reference_frame('panda_link0')

        

    def callback_rgbd(self, rgb, depth):
        with lock:
            self._update_rgb_image(rgb)
            self._update_depth_image(depth)

    def _update_rgb_image(self, msg):
        msg.__class__ = Image
        self.network_input['rgb'] = rnp.numpify(msg)
        self.network_input['msg_rgb'] = msg
        self.network_input['frame_stamp'] = msg.header.stamp

    def _update_depth_image(self, msg):
        msg.__class__ = Image
        raw_depth = rnp.numpify(msg)
        self.network_input['depth'] = raw_depth.astype(np.float32) / 1000.
        self.network_input['msg_depth'] = msg

    def _add_setup_scene(self):
        # add a large ground object to prevent collision with base tables
        # Todo: Hardcode other obstacles, e.g., controller box.
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "panda_link0"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = -0.05  # above the panda_hand frame
        box_name = "proxy_ground"
        self.planning_scene_interface.add_box(box_name, box_pose, size=(2.0, 2.0, 0.01))
        self.ground_name = box_name

        return self._wait_for_state_update(box_name, box_is_known=True, box_is_attached=True, timeout=8)

    def _wait_for_state_update(self, box_name, box_is_known=False, box_is_attached=False, timeout=4):
        # Same function from tutorial.
        scene = self.planning_scene_interface

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            is_known = box_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False

    def reset_robot(self):
        # set the robot to the default initial state
        # Todo: Determine a good initial state
        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[0] = 0.
        joint_goal[1] = - np.pi / 4.
        joint_goal[2] = 0.
        joint_goal[3] = -np.pi / 2.
        joint_goal[4] = 0.
        joint_goal[5] = np.pi / 3.
        joint_goal[6] = 0.

        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

        gripper_state = self.open_gripper()
        return self.move_group.get_current_joint_values(), \
            gripper_state

    def run_6d_grasp_inference(self):
        with lock:
            if not self._debug:
                np.save(f'{self.data_dir}/rgbd.npy', self.network_input)

        trigger = TriggerRequest()
        res = self.seg_service(trigger)
        print(res.message)
        if not res.success:
            return None
        res = self.grasp_service(trigger)
        print(res.message)
        if not res.success:
            return None
        result = np.load(f'{self.data_dir}/grasp.npy', allow_pickle=True).tolist()
        if len(result['grasps']) == 0:
            print("Warning. No grasp found.")
            return None
        return result

    def pose_at_grasp(self, grasp_pose_in_cam):
        curr_pose = from_pose_to_matrix(self.move_group.get_current_pose().pose)
        '''
            According to the paper, grasp pose is the frame of the panda_hand coordinated with a different axis, 
            i.e., 90 degree rotation along the z-axis.
        '''
        grasp_pose_in_base = self.tf_optical_to_base @ grasp_pose_in_cam  # T_{end-effector}{grasp}
        grasp_robot_hand_in_base = grasp_pose_in_base
        grasp_robot_hand_in_base = grasp_pose_in_base @ \
                                   np.array([[0, -1, 0, 0],
                                             [1, 0, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]])  # T_{panda_link0}{grasp_panda_hand_tf}
        grasp_pose = from_matrix_to_pose(grasp_robot_hand_in_base)

        return grasp_pose, grasp_robot_hand_in_base

    def go_to_grasp_pose(self, grasp_pose_in_cam, visual_configs):
        '''
            This function serves a example for the grasping demo.
            Todo: Add pre_grasp_pose, i.e., 5cm backward along the approaching direction.
            Todo: The full pipeline should be as follows:
            Todo: 1. Move to the pre_grasp_pose with full collision detection.
            Todo: 2. Disable collision detection between robot and octomap.
            Todo: 3. Move to grasp pose and grasp the object.
            Todo: 4. Lift the object by 20cm then move to the drop pose.
            Todo: 5. Drop the object and enable full collision detection.
        '''
        self.move_group.clear_pose_targets()
        command_grasp_pose, command_grasp_T = self.pose_at_grasp(grasp_pose_in_cam)
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.pose = command_grasp_pose
        # self.planning_scene_interface.add_box("pose_grip", box_pose, size=(0.01, 0.01, 0.01))
        
        self.move_group.set_pose_target(command_grasp_pose, end_effector_link='panda_hand')
        plan = self.move_group.plan()
        if not plan[0]:
            return False
        visualize_grasps(visual_configs['pc_full'], visual_configs['command_grasp'],
                         visual_configs['scores'],
                         pc_colors=visual_configs['pc_colors'])
        # scene = trimesh.Scene()
        # axis = trimesh.creation.axis()

        # handmesh = trimesh.load('../contact_graspnet/gripper_models/panda_gripper/hand.stl')
        # handmesh.apply_transform(command_grasp_T)
        # axis.apply_transform(command_grasp_T)
        # scene.add_geometry(axis)
        # scene.add_geometry(handmesh)

        # world_axis = trimesh.creation.axis()

        # scene.add_geometry(world_axis)
        # dmap_axis = trimesh.

        # scene.show()
        cmd = input("Execution? Press Enter to Execute.")  # press enter to execute
        if len(cmd) > 0:
            if cmd == 'end':
                exit(0)
            return False
        success = self.move_group.execute(plan[1], wait=True)
        if success:
            self.close_gripper()

        # Todo: Update here.
        self.reset_robot()

    def close_gripper(self, width=0.005):
        if self._debug:
            return None
        goal = franka_gripper.msg.GraspGoal()
        goal.width = width
        goal.epsilon.inner = 0.002
        goal.epsilon.outer = 0.002
        goal.speed = 0.1
        goal.force = 8
        self.robot_gripper_commander.send_goal(goal)
        self.robot_gripper_commander.wait_for_result()
        return self.robot_gripper_commander.get_result()

    def open_gripper(self, width=0.08):
        if self._debug:
            return None
        goal = franka_gripper.msg.GraspGoal()
        goal.width = width
        goal.epsilon.inner = 0.002
        goal.epsilon.outer = 0.002
        goal.speed = 0.1
        goal.force = 2.5
        self.robot_gripper_commander.send_goal(goal)
        self.robot_gripper_commander.wait_for_result()
        return self.robot_gripper_commander.get_result()

    def demo_6d_grasp(self):
        self.reset_robot()
        res = self.run_6d_grasp_inference()
        if res is None:
            return

        pc_full, pc_colors, scores, grasps = res['pc_full'], res['pc_colors'], res['scores'], res['grasps']
        # visualize_grasps(pc_full, grasps, scores, plot_opencv_cam=True, pc_colors=pc_colors)

        # sort all grasps and execute best ones, by default, demo the grasps on object 1.0
        # Todo: Update the logic here for fancy demos.
        demo_grasps = grasps[1.0][scores[1.0].argsort()[::-1]]
        for demo_grasp in demo_grasps:
            res['command_grasp'] = {1.0: np.expand_dims(demo_grasp, axis=0)}
            grasp_it = self.go_to_grasp_pose(demo_grasp, res)
            if grasp_it:
                break


if __name__ == '__main__':
    commander = Franka6DGraspCommand('/home/robotdev/franka_dev/ros_data', 'eye_to_hand', debug=False)
    time.sleep(1.)
    commander.demo_6d_grasp()
