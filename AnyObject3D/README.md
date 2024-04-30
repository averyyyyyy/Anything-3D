# Anything-3D: Towards Single-view Anything Reconstruction in the Wild

This repository contains a modified implementation of Anything-3D, a novel framework designed to address the challenges of 3D reconstruction from a single RGB image in unconstrained real-world scenarios. Our modified implementation uses a different segmentation model and alternative text prompt generation.

![pipeline](https://github.com/Anything-of-anything/Anything-3D/blob/main/AnyObject3D/assets/pipeline_anything3d.jpg)

## Preparation

Before running the code, make sure to install the required dependencies listed in the requirements.txt file. We modified the requirments.txt file to allow old packages to be installed from git. If you encounter issues, try cloning manually and doing a legacy install.

Running Environment
   ```bash 
   # we use cuda-11.8 runtime. and python 3.10
   python3.10 -m venv venv
   source venv/bin/activate #create virtual environment to prevent conflicts with system python
   cd /path/to/Anything-3D/AnyObject3D/src
   pip install -r 3DFuse/requirements.txt
   pip install -r requirements.txt
   ```
Pretrained models 
   ```bash
   cd /path/to/Anything-3D/AnyObject3D/src
   mkdir weights && cd weights 
   wget https://huggingface.co/jyseo/3DFuse_weights/resolve/main/models/3DFuse_sparse_depth_injector.ckpt
   ```

You may also need to create a directory called checkpoints
## Get 3D Objects

Now you can have a try to recontruct a 3D object from a single view image in the wild.
```bash 
# rename image with your desired object name and move it into the folder src/images/ 
# a simple demo to get a 3d car
mv car.jpg images/
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD/3DFuse:$PYTHONPATH
python ./main.py --image car \ # the name of your input image
                --point_coords 1000 1000 \ # point prompt in segment anything
                --seed 302651 # random seed 
```
The reconstruction process takes about 20 minutes on A10 GPU. After reconstruction, you can find the 3D car in the `results` folder. 
```bash
# find the resulted multi-view video 
find ./results -name '*.mp4'
```

## File organization
```
src/
├── 3DFuse/ # source code from 3DFuse
├── images/ # Images to be reconstructed
├── main.py # Our modified code
├── main_original.py #original code
├── mono_rec.py
├── rec_car.sh
└── requirements.txt
```
- `rec_car.sh` is a shell script that contains a sample code to reconstrct the `images/car.jpg`.
- `requirements.txt` specifies all enviroment requirements.

## License

This project is licensed under the [MIT License](https://github.com/Anything-of-anything/Anything-3D/blob/main/LICENSE).

## Acknowledgement
We express our gratitude to the exceptional project that inspired our code.
- [Segment-Anything](https://github.com/facebookresearch/segment-anything)
- [3DFuse](https://github.com/KU-CVLAB/3DFuse)
- [Point-E](https://github.com/openai/point-e)
- [BLIP](https://github.com/salesforce/LAVIS)
