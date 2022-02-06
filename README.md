# simple-accident-prediction
## Overview
This is an accident prediction model using GRU. The architecture of the model is developed with PyTorch. The input video frames are passed trough a CNN feature extractor (eg. ResNet50/VGG16), then the spatio-temporal relations of the extracted features are learnt using the GRU to predict the accident probablity of each video frame. The architecture is as below:
<div align=center>
  <img src="asset/architecture.PNG" alt="Architecture" width="800"/>
</div>
