# Hebrew ChatBot
In our Application We Are utilizing [Retrieval Augmented Generation (RAG)](https://arxiv.org/pdf/2005.11401) for querying and interacting with your data,
**either locally or deployed via cloud**.

![Screenshot](/images/hebrewChatBotExample.png)
![Demo of Hebrew ChatBot](/images/Video Hebrew Chat Bot.mp4))

We Are Giving The Option For Three LLM's 
- ChatGpt
- Dicta [Dicta](https://huggingface.co/dicta-il/dictalm2.0-instruct-GGUF)
- Aya   [Aya](https://huggingface.co/bartowski/aya-23-8B-GGUF)
#### We Strongly recommend these versions of files:
   - for Dicta - dictalm2.0-instruct.Q4_K_M.gguf
   - for Aya   - aya-23-8B-Q5_K_M.gguf
   - for ChatGpt You can Choose what ever you like  - we work with 'gpt-4o'
## Installation Instruction
you need to create a .env file with the Following Parameters:

| Environment Variable   | Value                                                      | Description                                                                       |
| ---------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------- |
| OPENAI_API_KEY         | open a count in OPENAI and get your key[openAI](https://platform.openai.com/docs/quickstart)| api key for ChatGpt|                                   |
|                        |                                                            |                                                                             |
| EMBEDDING_MODEL        | [Embedding Model](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)| embedding model for the Faiss DB Embedding Function  |
|                        |                                                            |                                                                             |
| MODEL_1                | the local path for dictalm2.0-instruct.Q4_K_M.gguf         | Dicta LLM with GGUF format of Tensor size Q4_K_M                            |
|                        |                                                            |                                                                             |
| MODEL_2                | the local path for aya-23-8B-Q5_K_M.gguf                   | Aya   LLM with GGUF format of Tensor size Q5_K_M                            |

#### To Implement - We recommend to use conda [conda](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html) and VScode [vscode](https://code.visualstudio.com/download)
#### Please follow the following steps:
- clone git@github.com:Eliyahou/HebrewChatBot.git
- cd HebrewChatBot
- conda create -n HebrewChatBot python=3.12.4 anaconda
- conda activate HebrewChatBot
- code .
- Install dependencies with `pip install -r requirements.txt`
  ## FOR CPU Installation
     - pip install llama-cpp-python on the terminal you use
  ## FOR GPU Installation - > the explanation is for CUDA 12.4 
     - Paste -> conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
     - Download your cuda version - [CUDA 12.4](https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_551.61_windows.exe)
     - Do the Propriate Installation in Visual Studio Installar ![installator](/images/installator.png)[download](https://visualstudio.microsoft.com/downloads/)
     - Take The Files from C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\visual_studio_integration\MSBuildExtensions and 
       Paste in C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations
       ### FOR Windows - In the Prompt of The Virtual Envroinment Paste:
          - $env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
          - $env:CUDATOOLKITDIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
          - pip install --force-reinstall --no-cache-dir llama-cpp-python
          - pip install numpy== 1.26.4
       ### For Linux - We Don't Think It Sude Be Very Diffrent
## TO Run The Application
  streamlit run rag/app.py  
 

