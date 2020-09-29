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


class PandaClosedLoopGraspController(object):
    """
    Perform closed-loop grasps from a single viewpoint using the Panda robot.
    """
    def __init__(self):
        ggrasp_service_name = '/ggrasp_service'
        rospy.wait_for_service(ggrasp_service_name + '/predict')
        self.ggrasp_srv = rospy.ServiceProxy(ggrasp_service_name + '/predict', GraspPrediction)

        self.curr_velocity_publish_rate = 100.0  # Hz
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.max_velo = 0.10
        self.velo_scale = 0.1

        self.initial_offset = 0.10
        self.gripper_width_offset = 0.03
        self.LINK_EE_OFFSET = self.robot_state.F_T_EE[14]
        
        self.curr_velo = Twist()
        self.best_grasp = Grasp()

        self.cs = ControlSwitcher({'velocity': 'cartesian_velocity_node_controller'})
        self.cs.switch_controller('velocity')
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

    def get_grasp(self):
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
        best_grasp.pose.position.z = best_grasp.pose.position.z - 0.055
        best_grasp.pose.position.z += self.initial_offset + self.LINK_EE_OFFSET # Offset from end effector position to

        return best_grasp
    
    def get_velocity(self):
        """Returns the distance from the target grasp from the current pose."""
        self.best_grasp = self.get_grasp()
        if not self.best_grasp:
            return False

        target_pose = self.best_grasp.pose
        current_pose = self.pc.get_current_pose()
        
        v = Twist()
        v.linear.x = target_pose.position.x - current_pose.position.x
        v.linear.y = target_pose.position.y - current_pose.position.y
        v.linear.z = target_pose.position.z - current_pose.position.z

        # v.angular.x,y = 0
        current_euler = tft.euler_from_quaternion(tfh.quaternion_to_list(current_pose.orientation))
        target_euler = tft.euler_from_quaternion(tfh.quaternion_to_list(target_pose.orientation))
        v.angular.z = target_euler[2] - current_euler[2]

        v.linear.x = self.velo_scale * v.linear.x
        v.linear.y = self.velo_scale * v.linear.y
        v.linear.z = self.velo_scale * v.linear.z
        v.angular.z = self.velo_scale * v.angular.z
                
        return v

    def __execute_grasp(self):
        self.cs.switch_controller('velocity')

        while self.robot_state.O_T_EE[-2] > self.best_grasp.pose.position.z and not any(self.robot_state.cartesian_contact) and not self.ROBOT_ERROR_DETECTED:            
            v = self.get_velocity()
            if not v:
                break
            else:
                self.curr_velo_pub.publish(v)
            rospy.sleep(0.01)

        self.best_grasp.pose.position.z -= self.initial_offset + self.LINK_EE_OFFSET
        self.cs.switch_controller('velocity')
        v = Twist()
        v.linear.z = -0.05
        while self.robot_state.O_T_EE[-2] > self.best_grasp.pose.position.z and not any(self.robot_state.cartesian_contact) and not self.ROBOT_ERROR_DETECTED:
            self.curr_velo_pub.publish(v)
            rospy.sleep(0.01)
            
        # Check for collisions
        if self.ROBOT_ERROR_DETECTED:
            return False

        # close the fingers.
        rospy.sleep(0.2)
        self.pc.grasp(0, force=1)

        best_grasp.pose.position.z += 0.2 # Raise robot arm by 10cm

        v.linear.z = 0.05
        while self.robot_state.O_T_EE[-2] < self.best_grasp.pose.position.z and not self.ROBOT_ERROR_DETECTED:
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
        self.__execute_grasp()
        # self.cs.switch_controller('velocity')
        # grasp_ret = self.__execute_best_grasp()
        # if not grasp_ret or self.ROBOT_ERROR_DETECTED:
        #     rospy.logerr('Something went wrong, aborting this run')
        #     if self.ROBOT_ERROR_DETECTED:
        #         self.__recover_robot_from_error()

if __name__ == '__main__':
    rospy.init_node('panda_closed_loop_grasp')
    pg = PandaClosedLoopGraspController()
    pg.go()
