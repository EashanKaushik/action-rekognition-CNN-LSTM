# action-rekognition-CNN-LSTM

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/210951344-a24d5669-1141-47c5-a68b-08ca2b106710.png" height="500px" width="500px"/>
</p>

Dataset: https://www.kaggle.com/datasets/eashankaushik/americansignlanguageactionrekognition

## Abstract
Action recognition is a challenging problem in deep learning. With the advancement in Deep Learning, ConvLSTM (8) and LSTMs have been widely used for action recognition. Both these approaches show promising results; in this project, we will implement an architecture by fusing these two architectures. We will also take advantage of the human skeletonbased approach in action recognition that takes advantage of a compact representation of human action.

## Introduction

American Sign Language (ASL) is widely spread within the deaf community as the primary source of communication with others, being used in 40 countries and containing more than 10,000 phrases. However, only 1% of the population knows sign language. Hence, recognizing American Sign Language (ASL) has various real-world applications. We have implemented a fused ConvLSTM and LSTM architecture in this project to detect 10 ASL signs. We successfully achieved a test accuracy of 0.901 on the custom ASL dataset. To compare our architecture with other Action Rekognition architecture, we have also trained our model on UCF-YouTube Action Dataset, achieving a test accuracy of 0.77. Human pose contains valuable information about ongoing human actions and can be combined with video frames or separately to detect actions. To detect and track human pose, we will make use of MediaPipe.

## Model

### LSTM
<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/210952084-690375ef-ea28-4f29-9074-504be220f1ef.png" />
</p>

### ConvLSTM
<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/210952011-ebbd5a85-fe17-42fa-bc00-a21a3d2dc585.png" />
</p>

### Fused Architecture
<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/210952148-21c7af79-6b10-495c-906b-035b1ed408b8.png" />
</p>

## Results
<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/210952204-49e155cd-bfec-4f70-9353-e6d3937b2a99.png" />
</p>

### Accuracy Curve
<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/210951664-d22caeb0-1a02-40be-b039-bbd51da6086c.PNG" />
</p>

### Loss Curve
<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/210951672-37a92aa4-256a-42a1-b381-71247ead0460.PNG" />
</p>

### Confusion Matrix
<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/210951689-9fb559f6-a265-4ee7-9dd6-f5d7ba991967.PNG" />
</p>
