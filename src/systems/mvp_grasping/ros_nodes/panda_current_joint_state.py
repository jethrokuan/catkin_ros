#! /usr/bin/env python
import rospy

from franka_control_wrappers.panda_commander import PandaCommander
import dougsm_helpers.tf_helpers as tfh
from dougsm_helpers.ros_control import ControlSwitcher

if __name__ == '__main__':
    rospy.init_node('panda_open_loop_grasp')
    cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                          'velocity': 'cartesian_velocity_node_controller'})
    cs.switch_controller('moveit')
    pc = PandaCommander(group_name='panda_arm', gripper=self.gripper)
    print(pc.get_current_pose())
