# rbtOCR

## Installation

```bash
# Install paddlepaddle
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda create -n rbtocr python=3.9
conda activate rbtocr
conda install paddlepaddle-gpu==2.1.0 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
#conda install paddlepaddle-gpu==2.1.0 cudatoolkit=11.2 -c conda-forge
conda install shapely numpy

# Check installation
python --version
python -c "import platform;print(platform.architecture()[0]);print(platform.machine());import paddle;print(paddle.utils.run_check())"

set KMP_DUPLICATE_LIB_OK=TRUE
```

(0.9706601466992665, ['030.jpg', '035.jpg', '092.jpg', '224.jpg', '228.jpg', '251.jpg', '254.jpg', '256.jpg', '257.jpg', '258.jpg', '278.jpg', '393.jpg'])

(0.9538152610441767, ['038.jpg', '039.jpg', '044.jpg', '070.jpg', '118.jpg', '119.jpg', '152.jpg', '196.jpg', '197.jpg', '202.jpg', '215.jpg', '221.jpg', '274.jpg', '275.jpg', '309.jpg', '334.jpg', '373.jpg', '388.jpg', '412.jpg', '414.jpg', '425.jpg', '427.jpg', '490.jpg'])

python tools/infer/predict_system.py --image_dir="./datasets/small/002.jpg" --det_model_dir="./inference/ch_ppocr_mobile_v1.1_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v1.1_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v1.1_cls_infer/" --use_gpu=True --gpu_mem=1000 --use_angle_cls=False --use_space_char=False