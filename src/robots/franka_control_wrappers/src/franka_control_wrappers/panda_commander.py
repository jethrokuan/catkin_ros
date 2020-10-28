import rospy
import actionlib

import moveit_commander
from moveit_commander.conversions import list_to_pose

import franka_gripper.msg
from franka_control_wrappers.panda_gripper import PandaGripper
from franka_control_wrappers.robotiq_gripper import RobotiqGripper
from franka_control.msg import ErrorRecoveryActionGoal


class PandaCommander(object):
    """
    PandaCommander is a class which wraps some basic moveit functions for the Panda Robot,
    and some via the panda API
    """
    def __init__(self, gripper="panda", group_name=None):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.groups = {}
        self.active_group = None
        self.set_group(group_name)
        self.saved_joint_poses = {}

        preset_joint_values = rospy.get_param("/panda_setup/saved_joint_values/")

        for name, joint_values in preset_joint_values.items():
            vs = [v for _, v in sorted(joint_values.items())]
            self.saved_joint_poses[name] = vs
            print("Loaded saved pose: {}".format(name))

        self.reset_publisher = rospy.Publisher('/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=1)

        if gripper == "panda":
            self.gripper = PandaGripper()
        elif gripper == "robotiq":
            self.gripper = RobotiqGripper()

    def save_current_pose(self, name):
        joint_values = self.active_group.get_current_joint_values()
        self.saved_joint_poses[name] = joint_values
        return

    def goto_saved_pose(self, name, velocity=1.0):
        joint_values = self.saved_joint_poses.get(name, None)
        if joint_values is None:
            raise ValueError("Cannot find saved pose: {}".format(name))
        self.goto_joints(joint_values, velocity=velocity)
            
    def print_debug_info(self):
        if self.active_group:
            planning_frame = self.active_group.get_planning_frame()
            print("============ Reference frame: %s" % planning_frame)
            eef_link = self.active_group.get_end_effector_link()
            print("============ End effector: %s" % eef_link)
        else:
            print("============ No active planning group.")
        print("============ Robot Groups:", self.robot.get_group_names())
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")

    def set_group(self, group_name):
        """
        Set the active move group
        :param group_name: move group name
        """
        self.active_group = group_name
        if group_name is None:
            self.active_group = None
            return
        else:
            if group_name not in self.groups:
                if group_name not in self.robot.get_group_names():
                    raise ValueError('Group name %s is not valid. Options are %s' % (group_name, self.robot.get_group_names()))
                self.groups[group_name] = moveit_commander.MoveGroupCommander(group_name)
            self.active_group = self.groups[group_name]

    def goto_joints(self, joint_values, velocity=1.0, group_name=None, wait=True):
        """
        Move to joint positions.
        :param joint_values:  Array of joint positions
        :param group_name:  Move group (use current if None)
        :param wait:  Wait for completion if True
        :return: Bool success
        """
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        joint_goal = self.active_group.get_current_joint_values()
        if len(joint_goal) != len(joint_values):
            raise IndexError('Expected %d Joint Values, got %d' % (len(joint_goal), len(joint_values)))
        for i, v in enumerate(joint_values):
            joint_goal[i] = v

        self.active_group.set_max_velocity_scaling_factor(velocity)
        success = self.active_group.go(joint_goal, wait)
        self.active_group.stop()
        return success

    def get_current_pose(self, group_name=None):
        """
        Returns the current pose of thet robot.
        """

        group = None
        if group_name:
            self.groups[group_name]
        else:
            group = self.active_group

        if not group:
            ValueError("Cannot find group")

        return group.get_current_pose().pose
        

    def goto_pose(self, pose, velocity=1.0, group_name=None, wait=True):
        """
        Move to pose
        :param pose: Array position & orientation [x, y, z, qx, qy, qz, qw]
        :param velocity: Velocity (fraction of max) [0.0, 1.0]
        :param group_name: Move group (use current if None)
        :param wait: Wait for completion if True
        :return: Bool success
        """
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        if type(pose) is list:
            pose = list_to_pose(pose)
        self.active_group.set_max_velocity_scaling_factor(velocity)
        self.active_group.set_pose_target(pose)
        success = self.active_group.go(wait=wait)
        self.active_group.stop()
        self.active_group.clear_pose_targets()
        return success

    def goto_pose_cartesian(self, pose, velocity=1.0, group_name=None, wait=True):
        """
        Move to pose following a cartesian trajectory.
        :param pose: Array position & orientation [x, y, z, qx, qy, qz, qw]
        :param velocity: Velocity (fraction of max) [0.0, 1.0]
        :param group_name: Move group (use current if None)
        :param wait: Wait for completion if True
        :return: Bool success
        """
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        if type(pose) is list:
            pose = list_to_pose(pose)

        self.active_group.set_max_velocity_scaling_factor(velocity)
        (plan, fraction) = self.active_group.compute_cartesian_path(
                                           [pose],   # waypoints to follow
                                           0.005,    # eef_step
                                           0.0)      # jump_threshold
        if fraction != 1.0:
            raise ValueError('Unable to plan entire path!')

        success = self.active_group.execute(plan, wait=wait)
        self.active_group.stop()
        self.active_group.clear_pose_targets()
        return success

    def goto_named_pose(self, pose_name, velocity=1.0, group_name=None, wait=True):
        """
        Move to named pos
        :param pose: Name of named pose
        :param velocity: Velocity (fraction of max) [0.0, 1.0]
        :param group_name: Move group (use current if None)
        :param wait: Wait for completion if True
        :return: Bool success
        """
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        self.active_group.set_max_velocity_scaling_factor(velocity)
        self.active_group.set_named_target(pose_name)
        success = self.active_group.go(wait=wait)
        self.active_group.stop()
        return success

    def stop(self):
        """
        Stop the current movement.
        """
        if self.active_group:
            self.active_group.stop()

    def recover(self):
        """
        Call the error reset action server.
        """
        self.reset_publisher.publish(ErrorRecoveryActionGoal())
        rospy.sleep(3.0)
