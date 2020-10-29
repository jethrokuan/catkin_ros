#! /usr/bin/env python

from __future__ import division, print_function

import rospy

import os
import time
import datetime
import numpy as np

from std_msgs.msg import Int16
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
import tf.transformations as tft

from franka_control_wrappers.panda_commander import PandaCommander

import dougsm_helpers.tf_helpers as tfh
from dougsm_helpers.ros_control import ControlSwitcher

from ggrasp.msg import Grasp


class PandaClosedLoopGraspController(object):
    """
    Perform closed-loop grasps from a single viewpoint using the Panda robot.
    """

    def __init__(self):
        self.gripper = rospy.get_param("~gripper", "panda")
        
        self.curr_velocity_publish_rate = 100.0  # Hz
        self.curr_velo_pub = rospy.Publisher(
            "/cartesian_velocity_node_controller/cartesian_velocity",
            Twist,
            queue_size=1,
        )
        self.grasp_sub = rospy.Subscriber(
            "/ggrasp/predict", Grasp, self.grasp_cmd_callback, queue_size=1
        )
        self.max_dist_to_target = 0.3  # distance to target to stop updating target pose
        self.linear_velo = 0.05

        self.curr_velo = Twist()
        self.best_grasp = Grasp()

        self.cs = ControlSwitcher(
            {
                "moveit": "position_joint_trajectory_controller",
                "velocity": "cartesian_velocity_node_controller",
            }
        )
        self.cs.switch_controller("velocity")
        self.pc = PandaCommander(group_name="panda_arm", gripper=self.gripper)

        self.robot_state = None
        self.ROBOT_ERROR_DETECTED = False
        self.BAD_UPDATE = False
        rospy.Subscriber(
            "/franka_state_controller/franka_states",
            FrankaState,
            self.__robot_state_callback,
            queue_size=1,
        )

    def grasp_cmd_callback(self, msg):
        best_grasp = msg

        tfh.publish_pose_as_transform(best_grasp.pose, "panda_link0", "G", 0.5)

        # Rotate quaternion by 45 deg on the z axis to account for home position being -45deg
        q_rot = tft.quaternion_from_euler(0, 0, np.pi / 4)
        q_new = tfh.list_to_quaternion(
            tft.quaternion_multiply(
                tfh.quaternion_to_list(best_grasp.pose.orientation), q_rot
            )
        )
        best_grasp.pose.orientation = q_new

        self.best_grasp = best_grasp

    def __recover_robot_from_error(self):
        rospy.logerr("Recovering")
        self.pc.recover()
        self.cs.switch_controller("moveit")
        self.pc.goto_saved_pose("start", velocity=0.1)
        rospy.logerr("Done")
        self.ROBOT_ERROR_DETECTED = False

    def __robot_state_callback(self, msg):
        self.robot_state = msg
        if any(self.robot_state.cartesian_collision):
            if not self.ROBOT_ERROR_DETECTED:
                rospy.logerr("Detected Cartesian Collision")
            self.ROBOT_ERROR_DETECTED = True
        for s in FrankaErrors.__slots__:
            if getattr(msg.current_errors, s):
                self.stop()
                if not self.ROBOT_ERROR_DETECTED:
                    rospy.logerr("Robot Error Detected")
                self.ROBOT_ERROR_DETECTED = True

    def dist_to_target(self, target_grasp):
        if target_grasp is None:
            return 100000 # large number when there is no target
        target_pose = target_grasp.pose
        current_pose = tfh.current_robot_pose("world", "panda_EE")
        x = target_pose.position.x - current_pose.position.x
        y = target_pose.position.y - current_pose.position.y
        z = target_pose.position.z - current_pose.position.z

        return np.sqrt(x ** 2 + y ** 2 + z ** 2)

    def get_velocity(self, target_pose):
        """Returns the distance from the target grasp from the current pose."""
        current_pose = tfh.current_robot_pose("world", "panda_EE")

        v = Twist()
        v.linear.x = target_pose.position.x - current_pose.position.x
        v.linear.y = target_pose.position.y - current_pose.position.y
        v.linear.z = target_pose.position.z - current_pose.position.z

        # v.angular.x,y = 0
        current_euler = tft.euler_from_quaternion(
            tfh.quaternion_to_list(current_pose.orientation)
        )
        target_euler = tft.euler_from_quaternion(
            tfh.quaternion_to_list(target_pose.orientation)
        )
        v.angular.z = target_euler[2] - current_euler[2]

        scaling_factor = self.linear_velo / np.sqrt(v.linear.x **2 + v.linear.y ** 2 + v.linear.z **2)
        v.linear.x = scaling_factor * v.linear.x
        v.linear.y = scaling_factor * v.linear.y
        v.linear.z = scaling_factor * v.linear.z
        
        v.angular.z = v.angular.z

        return v
    
    def __execute_grasp(self):
        target_grasp = None
        dist_to_target = self.dist_to_target(target_grasp) 
        gripper_width_offset = 0.01
        while (
            self.robot_state.O_T_EE[-2] > self.best_grasp.pose.position.z
            and not any(self.robot_state.cartesian_contact)
            and not self.ROBOT_ERROR_DETECTED
            and dist_to_target > 0.01
        ):
            if not self.best_grasp:
                break
            if dist_to_target > self.max_dist_to_target:
                target_grasp = self.best_grasp
                target_grasp.pose.position.z += 0.05
            v = self.get_velocity(target_grasp.pose)
            self.curr_velo_pub.publish(v)
            self.pc.gripper.set_gripper(target_grasp.width + gripper_width_offset)
            dist_to_target = self.dist_to_target(target_grasp)
                
            rospy.sleep(0.01)

        # Check for collisions
        if self.ROBOT_ERROR_DETECTED:
            return False

        target_grasp.pose.position.z -= 0.05
        
        while (
            not any(self.robot_state.cartesian_contact)
            and not self.ROBOT_ERROR_DETECTED
            and self.dist_to_target(target_grasp) > 0.01
        ):
            v = self.get_velocity(target_grasp.pose)
            self.curr_velo_pub.publish(v)
            rospy.sleep(0.01)


        # Check for collisions
        if self.ROBOT_ERROR_DETECTED:
            return False      

        rospy.sleep(1)
        self.cs.switch_controller("moveit")
        # close the fingers.
        rospy.sleep(0.2)
        self.pc.gripper.grasp(0, force=1)

        # Sometimes triggered by closing on something that pushes the robot
        if self.ROBOT_ERROR_DETECTED:
            return False

        return True

    def stop(self):
        self.pc.stop()
        self.curr_velo = Twist()
        self.curr_velo_pub.publish(self.curr_velo)

    def go(self):
        self.cs.switch_controller("moveit")
        self.pc.goto_saved_pose("start", velocity=0.1)
        self.pc.gripper.set_gripper(0.1)
        self.cs.switch_controller("velocity")
        grasp_ret = self.__execute_grasp()
        if not grasp_ret or self.ROBOT_ERROR_DETECTED:
            rospy.logerr("Something went wrong, aborting this run")
            if self.ROBOT_ERROR_DETECTED:
                self.__recover_robot_from_error()
        self.pc.goto_saved_pose("bin", velocity=0.1)
        self.pc.gripper.set_gripper(0.1)
        self.pc.goto_saved_pose("start", velocity=0.1)


if __name__ == "__main__":
    rospy.init_node("panda_closed_loop_grasp")
    pg = PandaClosedLoopGraspController()
    pg.go()
