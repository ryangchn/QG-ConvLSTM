# QG-ConvLSTM: Quality-Gated Convolutional LSTM for Enhancing Compressed Video
The project page of the paper:

Ren Yang, Xiaoyan Sun, Mai Xu and Wenjun Zeng, "Quality-Gated Convolutional LSTM for Enhancing Compressed Video", in IEEE International Conference on Multimedia and Expo (ICME), 2019.

# Test Codes
The code is to test our model on the sequence BasketballPass compressed by HEVC at QP = 42. The raw and compressed sequences can be dowbloaded from: 

https://drive.google.com/file/d/1l5HUCisQqezywCzdEKRXg0buarVlfdiG/view?usp=sharing

https://drive.google.com/file/d/1R0BNnOyJACCGzz1P2O-9Gc1JQIwR2jPq/view?usp=sharing

Please first download the raw and compressed sequences and than run test.py to evaluate.

Note that since the original codes were lost, these codes are re-implemented by the authors. The test results are very comparable with the reported numbers in the paper, but may be slightly different. For example, the PSNR improvement on BasketballPass at QP = 42 is 0.6061 dB (these codes) v.s. 0.6066 dB (Table 1 in the paper).  

# Contact
Email: r.yangchn@gmail.com

WeChat: yangren93
