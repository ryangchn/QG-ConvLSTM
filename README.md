# QG-ConvLSTM: Quality-Gated Convolutional LSTM for Enhancing Compressed Video
The project page of the paper:

Ren Yang, Xiaoyan Sun, Mai Xu and Wenjun Zeng, "Quality-Gated Convolutional LSTM for Enhancing Compressed Video", in IEEE International Conference on Multimedia and Expo (ICME), 2019.

https://arxiv.org/abs/1903.04596, or https://ieeexplore.ieee.org/document/8784864

# Code

## Test Code

This code is a demo to test our model on the sequences *BasketballPass* and *RaceHorses* compressed by HEVC at QP = 42. The raw and compressed sequences can be dowbloaded from: 

https://drive.google.com/file/d/1l5HUCisQqezywCzdEKRXg0buarVlfdiG/view?usp=sharing

https://drive.google.com/file/d/1R0BNnOyJACCGzz1P2O-9Gc1JQIwR2jPq/view?usp=sharing

https://drive.google.com/file/d/1ggmiu1N5CWZDR7b_qwgEdi0YLHc1nh2w/view?usp=sharing

https://drive.google.com/file/d/1qiW5T3QLKwGMH5VP9ds0z1UqSG7WqS9G/view?usp=sharing

Please first download the raw and compressed sequences and than run 

`python test.py --name BasketballPass --frames 500 --height 240 --width 416`

`python test.py --name RaceHorses --frames 300 --height 240 --width 416`

to evaluate.

The trained model for HEVC at QP = 37 is also available here.

## Generate feature.npy file

As discribed in the paper, we utilize quality-related features to generate the gates in ConvLSTM. In our code, feature.npy (e.g., BasketballPass_HEVC_QP42_quality_features.npy) contains a variable named 'feat', which is a matrix with shape [frame number, 38], i.e., each row is a 38-dimansion feature for one frame. 

The feat[frame number, 0:36] is obtained by the method of [1] with codes at 

http://live.ece.utexas.edu/research/quality/BRISQUE_release.zip. 

The feat[frame number, 36] and feat[frame number, 37] are the QP and total bits of the frame, respectively. 


[1] A. Mittal, A. K. Moorthy, and A. C. Bovik, “No-reference image quality assessment in the spatial domain,” IEEE Transactions on Image Processing, pp. 4695–4708, 2012.


# Contact
Email: r.yangchn@gmail.com
