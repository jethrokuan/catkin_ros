from franka_control_wrappers.gripper import BaseGripper
import actionlib
import control_msgs.msg
import rospy

from std_msgs.msg import Bool, Empty

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
        client = actionlib.SimpleActionClient('robotiq', control_msgs.msg.GripperCommandAction)
        client.wait_for_server()
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = width
        goal.command.max_effort = effort
        client.send_goal(goal)
        if wait:
            return client.wait_for_result()
        return True

    def grasp(self, width, speed=0.1, force=0.1):
        """
        Perform a grasp.
        """
        tactile = rospy.get_param("~tactile", "no")
        if tactile == "y":
            print("USING TACTILE FEEDBACK FOR GRASP")
            publisher = rospy.Publisher('/pid_enable', Bool, queue_size=10)
            publisher.publish(data=True)
            rospy.wait_for_message("/pid_done", Empty)
            return True
        else:
            return self.set_gripper(-0.01)
