from franka_control_wrappers.gripper import BaseGripper
import actionlib
import control_msgs.msg

class RobotiqGripper(BaseGripper):
    def home_gripper(self):
        """
        Home and initialise the gripper
        :return: Bool success
        """
        raise NotImplementedError

    def set_gripper(self, width, speed=0.1, effort=40, wait=True):
        """
        Set gripper width.
        :param width: Width in metres
        :param speed: Move velocity (m/s)
        :param wait: Wait for completion if True
        :return: Bool success
        """
        client = actionlib.SimpleActionClient('gripper/', control_msgs.msg.GripperCommandAction)
        client.wait_for_server()
        client.send_goal(control_msgs.msg.GripperCommandGoal(
            position=width,
            effort=effort
        ))
        return client.wait_for_result()

    def grasp(self, width, speed=0.1, force=0.1):
        """
        Perform a grasp.
        """
        return self.set_gripper(0)
