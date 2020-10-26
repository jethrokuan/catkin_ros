import tf2_ros
import dougsm_helpers.tf_helpers as tfh
import tf.transformations as tft
import geometry_msgs.msg as gmsg
import numpy as np


def correct_grasp(grasp):
    """Corrects the grasp pose given the gripper.

    This is because panda_link7 is in an awkward rotation, so the desired grasp
    pose needs to be modified.
    """
    p = gmsg.Pose()
    p.orientation.w = 1.0
    # panda_link8 is the EE link in moveit
    pose_diff = tfh.convert_pose(p, "panda_link8", "panda_EE")
    q_new = tfh.list_to_quaternion(
        tft.quaternion_multiply(tfh.quaternion_to_list(grasp.pose.orientation),
                                tfh.quaternion_to_list(pose_diff.orientation)))
    grasp.pose.orientation = q_new
    grasp.pose.position.x -= pose_diff.position.x
    grasp.pose.position.y -= pose_diff.position.y
    grasp.pose.position.z -= pose_diff.position.z

    return grasp
