# Super-Resolution PET Using A Very Deep Convolutional Neural Network
Tzu-An Song<sup>1</sup>, Samadrita Roy Chowdhury<sup>1</sup>, Fan Yang<sup>1</sup>, Joyita Dutta<sup>1</sup></br>
<sup>1</sup>Department of Electrical and Computer Engineering, University of Massachusetts Lowell, Lowell, MA, 01854 USA and co-affiliated with Massachusetts General Hospital, Boston, MA, 02114.

Positron emission tomography (PET) suffers from severe resolution limitations which reduce its quantitative accuracy. In this article, we present a super-resolution (SR) imaging technique for PET based on convolutional neural networks (CNNs). To facilitate the resolution recovery process, we incorporate high-resolution (HR) anatomical information based on magnetic resonance (MR) imaging. We introduce the spatial location information of the input image patches as additional CNN inputs to accommodate the spatially-variant nature of the blur kernels in PET. We compared the performance of shallow (3-layer) and very deep (20-layer) CNNs with various combinations of the following inputs: low-resolution (LR) PET, radial locations, axial locations, and HR MR. To validate the CNN architectures, we performed both realistic simulation studies using the BrainWeb digital phantom and clinical studies using neuroimaging datasets. For both simulation and clinical studies, the LR PET images were based on the Siemens HR+ scanner. Two different scenarios were examined in simulation: one where the target HR image is the ground-truth phantom image and another where the target HR image is based on the Siemens HRRT scanner - a high-resolution dedicated brain PET scanner. The latter scenario was also examined using clinical neuroimaging datasets. A number of factors affected relative performance of the different CNN designs examined, including network depth, target image quality, and the resemblance between the target and anatomical images. In general, however, all deep CNNs outperformed classical penalized deconvolution and partial volume correction techniques by large margins both qualitatively (e.g., edge and contrast recovery) and quantitatively (as indicated by three metrics: peak signal-to-noise-ratio, structural similarity index, and contrast-to-noise ratio).

Published in: IEEE Transactions on Computational Imaging

Page(s): 518 - 528

DOI: 10.1109/TCI.2020.2964229

The paper can be found [here](https://ieeexplore.ieee.org/document/8950375).

## Prerequisites

This code uses:

- Python 2.7
- Pytorch 0.4.0
- matplotlib 2.2.4
- numpy 1.16.4
- scipy 1.2.1
- NVIDIA GPU
- CUDA 8.0
- CuDNN 7.1.2

## Dataset

BrainWeb (Simulated Brain Database):
https://brainweb.bic.mni.mcgill.ca/brainweb/

Alzheimer’s Disease Neuroimaging Initiative (ADNI) (Clinical Database):
http://adni.loni.usc.edu/

## Notes

The codes will be moved to python 3.7.

## UMASS_LOWELL_BIDSLab
Biomedical Imaging & Data Science Laboratory

Lab's website:
http://www.bidslab.org/index.html


Email: TzuAn_Song(at)student.uml.edu, 
       TzuAn.Song(at)MGH.HARVARD.EDU, 
       alansoong200(at)gamil.com.
