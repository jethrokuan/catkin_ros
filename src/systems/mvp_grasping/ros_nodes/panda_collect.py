#! /usr/bin/env python

from __future__ import division, print_function

import rospy

import os
import time
import datetime
import numpy as np
import uuid

import cv2
import cv_bridge
bridge = cv_bridge.CvBridge()

from std_msgs.msg import Int16
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
import tf.transformations as tft

from franka_control_wrappers.panda_commander import PandaCommander

import dougsm_helpers.tf_helpers as tfh
from dougsm_helpers.ros_control import ControlSwitcher

from ggrasp.msg import Grasp
from ggrasp.srv import GraspPrediction

from mvp_grasping.panda_base_grasping_controller import Logger, Run, Experiment

Run.log_properties = ['success', 'time', 'quality']
Experiment.log_properties = ['success_rate', 'mpph']


class PandaCollectController(object):
    """
    Perform open-loop grasps from a single viewpoint using the Panda robot.
    Collect datapoints based on whether grasps are successful.
    """
    def __init__(self):
        gripper = rospy.get_param("~gripper", "panda")
        if gripper == "panda":
            self.LINK_EE_OFFSET = 0.1384
        elif gripper == "robotiq":
            self.LINK_EE_OFFSET = 0.245

        ggrasp_service_name = '/ggrasp'
        rospy.wait_for_service(ggrasp_service_name + '/predict')
        self.ggrasp_srv = rospy.ServiceProxy(ggrasp_service_name + '/predict', GraspPrediction)
        
        self.save_dir = "./ggrasp_data/"
        self.depth_img = None
        self.grasp = None
        self.curr_velocity_publish_rate = 100.0  # Hz
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.max_velo = 0.10
        self.curr_velo = Twist()
        self.best_grasp = Grasp()

        self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                                   'velocity': 'cartesian_velocity_node_controller'})
        self.cs.switch_controller('moveit')
        self.pc = PandaCommander(group_name='panda_arm', gripper=gripper)

        self.initial_pose = None
        self.robot_state = None
        self.ROBOT_ERROR_DETECTED = False
        self.BAD_UPDATE = False
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

    def __recover_robot_from_error(self):
        rospy.logerr('Recovering')
        self.pc.recover()
        self.cs.switch_controller('moveit')
        self.pc.goto_pose(self.initial_pose, velocity=0.1)
        rospy.logerr('Done')
        self.ROBOT_ERROR_DETECTED = False

    def __robot_state_callback(self, msg):
        self.robot_state = msg
        if any(self.robot_state.cartesian_collision):
            if not self.ROBOT_ERROR_DETECTED:
                rospy.logerr('Detected Cartesian Collision')
            self.ROBOT_ERROR_DETECTED = True
        for s in FrankaErrors.__slots__:
            if getattr(msg.current_errors, s):
                self.stop()
                if not self.ROBOT_ERROR_DETECTED:
                    rospy.logerr('Robot Error Detected')
                self.ROBOT_ERROR_DETECTED = True

    def __execute_best_grasp(self):
            self.cs.switch_controller('moveit')

            ret = self.ggrasp_srv.call()

            if not ret.success:
                return False
            
            self.best_grasp = ret.best_grasp
            best_grasp = self.best_grasp
            self.depth_img = bridge.imgmsg_to_cv2(ret.depth)
            self.grasp = ret.grasp

            tfh.publish_pose_as_transform(self.best_grasp.pose, 'panda_link0', 'G', 0.5)

            # Offset for initial pose.
            initial_offset = 0.05
            gripper_width_offset = 0.03

            # Add some limits, plus a starting offset.
            best_grasp.pose.position.z += initial_offset + self.LINK_EE_OFFSET  # Offset from end effector position to

            self.pc.gripper.set_gripper(best_grasp.width + gripper_width_offset, wait=False)
            rospy.sleep(0.1)
            self.pc.goto_pose(best_grasp.pose, velocity=0.1)

            # Reset the position
            best_grasp.pose.position.z -= initial_offset + LINK_EE_OFFSET

            self.cs.switch_controller('velocity')
            v = Twist()
            v.linear.z = -0.05

            # Monitor robot state and descend
            while self.robot_state.O_T_EE[-2] > best_grasp.pose.position.z and not any(self.robot_state.cartesian_contact) and not self.ROBOT_ERROR_DETECTED:
                self.curr_velo_pub.publish(v)
                rospy.sleep(0.01)
            
            # Check for collisions
            if self.ROBOT_ERROR_DETECTED:
                return False

            rospy.sleep(1)
            self.cs.switch_controller('moveit')
            # close the fingers.
            rospy.sleep(0.2)
            self.pc.gripper.grasp(0, force=1)
            self.pc.goto_pose(self.initial_pose, velocity=0.1)
            
            # Sometimes triggered by closing on something that pushes the robot
            if self.ROBOT_ERROR_DETECTED:
                return False
            
            return True

    def stop(self):
        self.pc.stop()
        self.curr_velo = Twist()
        self.curr_velo_pub.publish(self.curr_velo)

    def go(self):
        self.initial_pose = self.pc.get_current_pose()
        self.cs.switch_controller('moveit')
        self.pc.gripper.set_gripper(0.1)

        self.cs.switch_controller('velocity')
        grasp_ret = self.__execute_best_grasp()
        if not grasp_ret or self.ROBOT_ERROR_DETECTED:
            rospy.logerr('Something went wrong, aborting this run')
            if self.ROBOT_ERROR_DETECTED:
                self.__recover_robot_from_error()
        self.pc.gripper.set_gripper(0.1)

        successful = raw_input("grasp successful? (y/n): ")
        if successful == "y":
            UID = uuid.uuid1()
            cv2.imwrite(self.save_dir + str(UID) + ".png", self.depth_img)
            with open(self.save_dir + str(UID) + ".grasp", "w") as f:
                f.write(";".join([str(x) for x in self.grasp]))

if __name__ == '__main__':
    rospy.init_node('panda_collect')
    pg = PandaCollectController()
    pg.go()
