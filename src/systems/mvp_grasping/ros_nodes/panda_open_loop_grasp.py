#! /usr/bin/env python

from __future__ import division, print_function

import rospy

import os
import time
import datetime
import numpy as np


from std_msgs.msg import Int16
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


class PandaOpenLoopGraspController(object):
    """
    Perform open-loop grasps from a single viewpoint using the Panda robot.
    """
    def __init__(self):
        ggrasp_service_name = '/ggrasp_service'
        rospy.wait_for_service(ggrasp_service_name + '/predict')
        self.ggrasp_srv = rospy.ServiceProxy(ggrasp_service_name + '/predict', GraspPrediction)

        self.curr_velocity_publish_rate = 100.0  # Hz
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.max_velo = 0.10
        self.curr_velo = Twist()
        self.best_grasp = Grasp()

        self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                                   'velocity': 'cartesian_velocity_node_controller'})
        self.cs.switch_controller('moveit')
        self.pc = PandaCommander(group_name='panda_arm')

        self.robot_state = None
        self.ROBOT_ERROR_DETECTED = False
        self.BAD_UPDATE = False
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

    def __recover_robot_from_error(self):
        rospy.logerr('Recovering')
        self.pc.recover()
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
            best_grasp = ret.best_grasp
            self.best_grasp = best_grasp

            tfh.publish_pose_as_transform(best_grasp.pose, 'panda_link0', 'G', 0.5)

            # Rotate quaternion by 45 deg on the z axis to account for home position being -45deg
            q_rot = tft.quaternion_from_euler(0, 0, np.pi/4)
            q_new = tfh.list_to_quaternion(tft.quaternion_multiply(tfh.quaternion_to_list(best_grasp.pose.orientation), q_rot))
            best_grasp.pose.orientation = q_new

            print(best_grasp)
            
            if raw_input('Continue?') == '0':
                return False

            # Offset for initial pose.
            initial_offset = 0.10
            LINK_EE_OFFSET = self.robot_state.F_T_EE[14]

            # Add some limits, plus a starting offset.
            best_grasp.pose.position.z = best_grasp.pose.position.z - 0.055
            best_grasp.pose.position.z += initial_offset + LINK_EE_OFFSET  # Offset from end effector position to

            self.pc.set_gripper(best_grasp.width, wait=False)
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

            # close the fingers.
            rospy.sleep(0.2)
            self.pc.grasp(0, force=2)

            best_grasp.pose.position.z += 0.2 # Raise robot arm by 10cm

            v.linear.z = 0.05
            while self.robot_state.O_T_EE[-2] < best_grasp.pose.position.z and not self.ROBOT_ERROR_DETECTED:
                self.curr_velo_pub.publish(v)
                rospy.sleep(0.01)

            v.linear.z = 0
            self.curr_velo_pub.publish(v)
            self.pc.set_gripper(0.1)
            
            # Sometimes triggered by closing on something that pushes the robot
            if self.ROBOT_ERROR_DETECTED:
                return False
            
            return True

    def stop(self):
        self.pc.stop()
        self.curr_velo = Twist()
        self.curr_velo_pub.publish(self.curr_velo)

    def go(self):
        self.cs.switch_controller('moveit')
        self.pc.set_gripper(0.1)

        self.cs.switch_controller('velocity')
        grasp_ret = self.__execute_best_grasp()
        if not grasp_ret or self.ROBOT_ERROR_DETECTED:
            rospy.logerr('Something went wrong, aborting this run')
            if self.ROBOT_ERROR_DETECTED:
                self.__recover_robot_from_error()

if __name__ == '__main__':
    rospy.init_node('panda_open_loop_grasp')
    pg = PandaOpenLoopGraspController()
    pg.go()
