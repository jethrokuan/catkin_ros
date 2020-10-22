# Grasping Pipeline

This is a simple grasping pipeline implemented on the Franka Emika Panda robot.
Currently, two grippers are supported: the Franka gripper and the Robotiq 2F
gripper series.

![[Franka Grasp](https://j.gifs.com/L7op6r.gif)](https://www.youtube.com/watch?v=5qAasB84R9E)

Depth images are obtained via an Intel Realsense camera, and passed to a neural
network to obtain the appropriate grasp point. In the open-loop scenario, the
grasp point is obtained once, upon network run. In the closed-loop scenario,
depth images are fed to the ggrasp network in real-time, recomputing the target
pose.

This depends on `algorithms/ggrasp`, and the `robots/franka` related libraries.

# Controllers

## Open-loop

    roslaunch mvp_grasping wrist_realsense.launch # Setup realsense camera
    roslaunch ggrasp ggrasp_rt.launch             # Start real-time prediction
    roslaunch mvp_grasping robot_bringup.launch   # Start robot controller
    rosrun mvp_grasping panda_open_loop_grasp.py  # Run open-loop scenario

1.  `ggrasp_rt` continuously receives depth images from the intel realsense, and
    produces target grasp poses (at `/ggrasp/predict`)
2.  A single target grasp pose is read, and used
3.  The robot moves to a pregrasp pose `0.05m` above the target pose, and
    performs a descent using velocity control to grasp the object
4.  The robot then returns to its initial position and releases the gripper

## Closed-loop

    roslaunch mvp_grasping wrist_realsense.launch # Setup realsense camera
    roslaunch ggrasp ggrasp_rt.launch             # Start real-time prediction
    roslaunch mvp_grasping robot_bringup.launch   # Start robot controller
    rosrun mvp_grasping panda_closed_loop_grasp.py  # Run closed-loop scenario

1.  `ggrasp_rt` continuously receives depth images from the intel realsense, and
    produces target grasp poses (at `/ggrasp/predict`). Some additional machinery
    is in place to ensure that the target grasp pose does not jump too wildly.
2.  Target grasp poses are continuously read from the `/ggrasp/predict` stream.
3.  While the distance of the robot&rsquo;s current pose is > `max_dist_to_target`, the
    robot uses velocity control, with velocity vector \(\lambda \times
       (T_{target} - T_{current})\).


# Running with the Robotiq gripper

Replace the commands:

    roslaunch mvp_grasping robot_bringup.launch

with:

    roslaunch mvp_grasping robot_bringup.launch gripper:=robotiq

and the `rosrun` commands e.g.:

    rosrun mvp_grasping panda_closed_loop_grasp.py

with:

    rosrun mvp_grasping panda_closed_loop_grasp.py _gripper:=robotiq

## Data Collection

    roslaunch mvp_grasping wrist_realsense.launch # Setup realsense camera
    roslaunch ggrasp ggrasp_rt.launch             # Start real-time prediction
    roslaunch mvp_grasping robot_bringup.launch   # Start robot controller
    rosrun mvp_grasping panda_collect.py          # Run data collection

The idea is to collect data on previously unseen objects, bootstrapping using a
network trained on a public dataset.

A target pose is obtained via the network, and executed using the open-loop
control scenario. If the grasp is a success, the depth image is saved, along
with the grasp bounding box given by the predicted target pose. The data is
saved in a format similar to the Jacquard dataset, and can be used for
fine-tuning the model on relevant objects.

# Notes

## Depth Image Filtering

The Intel Realsense node contains multiple filters for post-processing of the
depth image. By default, the temporal filter is turned on. This filter produces
**poor** results for the closed-loop grasping scenario, as the post-processed
depth image using a temporal filter with a moving camera is very poor. Ensure
that the temporal filter is turned off.

## Clearing the Octomap

To prevent moveit from erroring, claiming that the end-effector is in contact
with the object it has grasped, we need to clear the octomap.

Alternatively, one should instead disable octomaps altogether, and manually add
the planar table to the collision scene. This has the downside that moveit
cannot plain collision avoidance against objects.
