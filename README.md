## How to install tenserflow
- Install anaconda
- CUDA Toolkit and driver version: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html/
- tensorflow cuda cudnn version
https://www.tensorflow.org/install/source#gpu
- install tensorflow-gpu, cuda, and cudnn via conda
    ```bash  
    # create env
    conda create -n tf2 python=3.6
    # install tensorflow 2
    conda install tensorflow-gpu=2.1
    # or try this
    conda install cudatoolkit==10.1.243
    conda install cudnn==7.6.5        
    pip install tensorflow-gpu==2.1.0
    # delete env
    conda env remove -n tf2
    ```
## jupyter lab
```bash
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --no-browser --port=8787
```