#! /usr/bin/env python

from __future__ import division, print_function

import rospy

import os
import time
import datetime


from std_msgs.msg import Int16
from geometry_msgs.msg import Twist
from franka_msgs.msg import FrankaState, Errors as FrankaErrors

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
        gripper = rospy.get_param("~gripper", "panda")
        self.curr_velocity_publish_rate = 100.0  # Hz
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.max_velo = 0.10
        self.curr_velo = Twist()
        self.best_grasp = Grasp()

        self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                                   'velocity': 'cartesian_velocity_node_controller'})
        self.cs.switch_controller('moveit')
        self.pc = PandaCommander(group_name='panda_arm', gripper=gripper)

        self.robot_state = None
        self.ROBOT_ERROR_DETECTED = False
        self.BAD_UPDATE = False
        self.initial_pose = None
        
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

    def stop(self):
        self.pc.stop()
        self.curr_velo = Twist()
        self.curr_velo_pub.publish(self.curr_velo)

    def go(self):
        self.cs.switch_controller('moveit')
        print(self.initial_pose)
        self.pc.goto_pose()
        new_pose = self.initial_pose
        new_pose.position.z += 0.05
        self.pc.goto_pose(new_pose, velocity=0.1)
        rospy.sleep(2)
        self.pc.goto_pose(initial_pose, velocity=0.1)
        self.pc.gripper.set_gripper(0.1)
        rospy.sleep(2)
        self.pc.gripper.set_gripper(1)

if __name__ == '__main__':
    rospy.init_node('panda_test_grasp')
    pg = PandaOpenLoopGraspController()
    pg.go()
