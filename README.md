# Hebrew ChatBot
![Screenshot](/images/hebrewChatBotExample.png)

To Use the system - We recommand to use conda
Please follow the following steps:
clone git@github.com:Eliyahou/HebrewChatBot.git
# Paste The Following in your Prompt
cd HebrewChatBot
conda create -n HebrewChatBot python=3.12.4 anaconda
conda activate HebrewChatBot
code .
Install dependencies with `pip install -r requirements.txt`
## For CPU
if you use cpu ->pip install llama-cpp-python on the terminal you use
# for GPU Installation - > the explanation is for CUDA 12.4 
Download your cuda version - [CUDA 12.4](https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_551.61_windows.exe)
Do the Propriate Installation in Visual Studio Installar ![installator](/images/installator.png)
Take The Files from C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\visual_studio_integration\MSBuildExtensions and
Paste in C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations
## For Windows
  1. In the Prompt of The Virtual Envroinment paste-> $env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
  2. paste-> $env:CUDATOOLKITDIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
  3. pip install --force-reinstall --no-cache-dir llama-cpp-python
  4. pip install numpy== 1.26.4
 

