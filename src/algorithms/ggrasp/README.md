[![Real-time Grasp Prediction](https://j.gifs.com/wVBLXm.gif)](https://youtu.be/yYmkOf3FZQ8)

This sub-project contains the code for training and deploying two models:

1.  [Generative Grasping CNN (GGCNN)](https://github.com/dougsm/ggcnn)

> GG-CNN is a lightweight, fully-convolutional network which predicts the quality
> and pose of antipodal grasps at every pixel in an input depth image. The
> lightweight and single-pass generative nature of GG-CNN allows for fast
> execution and closed-loop control, enabling accurate grasping in dynamic
> environments when objects are moved during the grasp attempt.

1.  [GR-ConvNet](https://github.com/skumra/robotic-grasping)

> GR-ConvNet is a novel generative residual convolutional neural network based
> model architecture which detects objects in the camera’s field of view and
> predicts a suitable antipodal grasp configuration for the objects in the image.

# Motivation

> This is a brief introduction to the network, for more details, read the original
> papers.

Many of the prior art on grasping rely on complex pipelines that take a long
time to execute. For example, DexNet uses a network to generate grasp candidates
from point clouds, and aseparate network to rank these grasp candidates to
select the best grasp. These pipelines are slow to execute, hence have to be
excuted open-loop. These open-loop grasps require careful calibration of the
robot and the sensors.

In contrast, the generative grasping networks here directly generate a dense
prediction of antipodal grasp poses, and a quality measure for every pixel in
the input depth image. These networks can produce predictions in real time,
allowing for closed-loop control.

# Obtaining the Training Data

The networks can be trained on two datasets:

1.  [The Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php)
2.  [The Jacquard Dataset](https://jacquard.liris.cnrs.fr/)

# Modifications to the Original Works

The GG-CNN2 paper uses a per-pixel MSE loss. In other works on depth estimation
and optical flow, it is more common to use a smooth L1 loss. This was used in
the GR-ConvNet paper. We have also found the smooth L1 loss to result in better
performance.

In addition, we introduce a smoothness regularization term. The intuition here
is that grasp poses in the local neighbourhood should be similar in quality, and
in their values (angle, width). This smoothness loss is computed as the L1 norm
of the second-order gradients for the output images. We find that the smoothness
loss not only significantly improves the IOU metric we use, but also improves
grasp stability when applied to our grasp controllers.

# Training the Models

To train the models, we use guild.ai. First, navigate to the `src` folder (which
contains the guild.yml).

Run `guild operations` to see a list of operations. For example, to train the ggcnn2 model:

    guild run ggcnn2:train loss:smooth_l1_loss smoothness_weight:0.5
    dataset:JacquardDataset dataset-path:/path/to/dataset

Sample trained models can be found in the `saved_models` folder.

