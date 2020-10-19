class BaseGripper(object):
    def home_gripper(self):
        """
        Home and initialise the gripper
        :return: Bool success
        """
        raise NotImplementedError

    def set_gripper(self, width, speed=0.1, effort=40, wait=True):
        """
        Set gripper with.
        :param width: Width in metres
        :param speed: Move velocity (m/s)
        :param wait: Wait for completion if True
        :return: Bool success
        """
        raise NotImplementedError

    def grasp(self, width, speed=0.1, force=0.1):
        """
        Perform a grasp.
        """
        raise NotImplementedError
