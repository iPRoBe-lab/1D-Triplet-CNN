
1D-Triplet-CNN
===============================

PyTorch implementation of the 1D-Triplet-CNN neural network model described in *Fusing MFCC and LPC Features using 1D Triplet CNN for Speaker Recognition in Severely Degraded Audio Signals* by A. Chowdhury, and A. Ross.

## Research Article

[Anurag Chowdhury](https://github.com/ChowdhuryAnurag), and [Arun Ross](http://www.cse.msu.edu/~rossarun/) (2019) *Fusing MFCC and LPC Features using 1D Triplet CNN for Speaker Recognition in Severely Degraded Audio Signals* IEEE Transactions on Information Forensics and Security (2019).   

- IEEE Xplore: [https://ieeexplore.ieee.org/document/8839817](https://ieeexplore.ieee.org/document/8839817)

![1D-Triplet-CNN Model](/images/arch.png)
Format: ![Alt Text](url)

![1D-Triplet-CNN Details](/images/Feature_fusion.png)
Format: ![Alt Text](url)

## Implementation details and requirements


The model was implemented in PyTorch 1.2.1 using Python 3.6 and may be compatible with different versions of PyTorch and Python, but it has not been tested.

Additional requirements are listed in the [./requirements.txt](./requirements.txt) file. 


## Usage

**Source code and model parameters**

The source code of the 1D-Triplet-CNN model can be found in the [src](./src) subdirectory, and a pre-trained model is available in the [trained_models](./trained_models) subdirectory.


**Dataset**

The pre-trained model avilable in the [model](./model) subdirectory was trained on a subset of Fisher speech corpus obtained from https://catalog.ldc.upenn.edu/LDC2004S13. The training data was also degraded with varying degrees of Babble noise obtained from [NOISEX-92](http://www.speech.cs.cmu.edu/comp.speech/Section1/Data/noisex.html) dataset.

**Training the 1D-Triplet-CNN model**

In order to train a 1D-Triplet-CNN model as described in the research paper, Use the 1D-Triplet-CNN implementation given in [models](./models) subdirectory.
The network attains optimal performance when trained using a triplet learning framework. Read the research paper for more details on training the model.

*Testing with the pretrained model*

*Recommended audio specifications*
Usually, 2 secs of audio sampled at 8000KHz is enough to produce reliable speaker recognition results.
Longer audio samples will make the recognition task significantly slower with no significant benefits to performance.
Audio samples smaller than 1secs with have considerable performance loss.

*Usage*
1. Satisfy the requirements listed in the [./requirements.txt](./requirements.txt) file. 
2. Run [src/extractFeatures.m](src/extractFeatures.m) in MATLAB R2019a(or newer) to extract MFCC-LPC features from audio files placed in [sample_audio](sample_audio) subdirectory
3. and save corresponding features as individual .mat files in [sample_feature](sample_feature) subdirectory.
4. Run [src/extractFeatures.m](src/test.py) in Python 3.6 to evaluate some sample audio pairs for generating speaker verification scores.

**Examples**
Some usage examples might be added in future.