# MIFFuse

This code is for "MIFFuse: A Multi-level Feature Fusion Network for Infrared and visible Images"



## Requirements

1. Python 3.6

2. Pytorch 1.7.0

3. CUDA 11.0
4. The images needed for the training process can be downloaded [here](https://1drv.ms/u/s!Ak33bhBC1gcvgaktJUQV7sMoOiqXAw?e=KGIOqH) (only 10 sample images are included in the project)

## Code Structure

 - train: Training code  on the dataset.
 - test: Test the model on TNO/CVC_14 dataset
 - test_RGB: Test the model on Flir dataset
 - MIF_net: MIFFufuse architecture  

## Citation
```
@ARTICLE{IPLF,  
author={Zhu, Depeng and Zhan, Weida and Jiang, Yichun and Xiaoyu, Xu and Guo, Renzhong},  
journal={IEEE Sensors Journal},   
title={IPLF:A Novel Image Pair Learning Fusion Network for Infrared and Visible Image},   
year={2022},  
volume={},  
number={},  
pages={1-1},  
doi={10.1109/JSEN.2022.3161733}}
```
  
If you have any question, please email to me <zhudepeng@mails.cust.edu.cn>
