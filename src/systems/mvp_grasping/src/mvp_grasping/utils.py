import tf.transformations as tft
import dougsm_helpers.tf_helpers as tfh
import numpy as np

def correct_grasp(grasp, gripper):
    """Corrects the grasp pose given the gripper.

    This is because panda_link7 is in an awkward rotation, so the desired grasp
    pose needs to be modified.
    """
    if gripper == "robotiq":
        angle = np.pi/2
    elif gripper == "panda":
        angle = np.pi/4
    else:
        raise ValueError("Unsupported gripper: {}".format(gripper))

    q_rot = tft.quaternion_from_euler(0, 0, angle)
    q_new = tfh.list_to_quaternion(tft.quaternion_multiply(tfh.quaternion_to_list(grasp.pose.orientation), q_rot))
    grasp.pose.orientation = q_new

    return grasp
