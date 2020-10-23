This catkin workspace is organized like a mono-repo, and will be used for all
robotics projects.

> The submodule commits have been carefully chosen to match the
> robotics setup at the CLEAR-NUS lab. DO NOT UPGRADE UNNECESSARILY.

The workspace is organized as follows:

-   **algorithms:** contains packages that process data and produce output. Typically, these packages use traditional techniques, or wrap neural networks that consume sensor data and output predictions.
-   **robots:** contains packages relevant to supporting the robots we have in the lab (e.g. the Franka Emika Panda)
-   **sensors:** contains packages for sensor support (e.g. RGBD cameras, tactile sensors)
-   **utils:** contains any utilities that may be used throughout other packages
-   **systems:** for a particular experimental setup, a certain system might be use a different combination of sensors, robots and algorithms. Each system will be a catkin package that depends on packages, and contain system-specific code.

Packages should be written targeting Python 3, and newer versions of ROS (Noetic and above). Documentation for key systems can be found here:

1.  [Grasping with the Franka Emika and Franka/Robotiq Grippers](src/systems/mvp_grasping/README.md)
2.  [Obtaining the transform between the camera and robot using Apriltag](docs/apriltag_transform.md)

# Setup

To ensure that every project in this repository can run, perform the following steps sequentially.

    cd ~
    git clone https://github.com/jethrokuan/catkin_ros/ --recursive
    cd src/robots/libfranka
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cmake --build .
    cd ~/catkin_ros/
    rosdep install --from-paths src --ignore-src src -r -y --skip-keys libfranka
    catkin_make_isolated -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=~/catkin_ros/src/robots/libfranka/build


# Future Tasks

## TODO Update MoveIt! to latest (Melodic/Noetic/ROS 2)

Current MoveIt installed in Kinetic is missing some planners that the new `panda_moveit_config` uses.

## TODO Move to colcon build system

## TODO Build Robotiq 2F Joint State Publisher

Official Robotiq ROS package does not support this.

# Reference Documentation

## TF

In Rviz, the axes colors are as follows:

-   **X axis:** red
-   **Y axis:** green
-   **Z axis:** blue


## Cameras

### Frames

The `camera_link` frame is a center point for all other link frames to relate to. In the D400 series, this is defined as the `camera_depth_frame`. To verify, we can run:

    rosrun tf tf_echo camera_depth_frame camera_link

We should see:

    At time 0.000
    - Translation: [0.000, 0.000, 0.000]
    - Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]
                in RPY (radian) [0.000, -0.000, 0.000]
                in RPY (degree) [0.000, -0.000, 0.000]

Image data is usually published in the optical frame of the camera. In practice, this means that to obtain camera data in another base frame,

### Realsense

1.  Camera Calibration

### Image Processing

[depth_image_proc](https://wiki.ros.org/depth_image_proc) is a library that processes depth images produced by cameras such as the realsense.

`depth_image_proc/convert_metric` converts raw uint16 images in mm to float depth image in m.


## AprilTag

[AprilTag](https://april.eecs.umich.edu/software/apriltag) is a visual fiducial system that is useful for multiple robotics tasks, including camera calibration. It uses simple printed targets that look like QR codes. The [AprilTag ROS](http://wiki.ros.org/apriltag_ros) library is able to establish the tf transform betwen the tag and the camera.

    rosrun apriltag_ros continuous_detection.launch

To establish a transform between a camera and a base frame (e.g. a robot), we place the april tag at a known displacement (and rotation) from the robot frame. To publish a static transfrom, we use:

    rosrun tf static_tf_publisher ...

We use the AprilTag to compute the transform from the camera to the tag.

We can then do:

    rosrun tf tf_echo camera_link robot_frame

And save this transform for later use.
