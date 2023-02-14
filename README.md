# ADNet: Rethinking the Shrunk Polygon-Based Approach in Scene Text Detection

## Introduction
This is a pytorch implementation for paper [ADNet](https://ieeexplore.ieee.org/document/9927333) (TMM2022). ADNet is a shrunk polygon-based scene text detector, which uses an instance-wise dilation factor to obtain more complete and tight results. This repository is built on [DBNet](https://github.com/MhLiao/DB).

## ToDo List

- [x] Release code
- [x] Document for Installation
- [x] Trained models
- [x] Document for testing and training
- [x] Evaluation

## Installation

### Requirements:
- Python==3.7
- Pytorch==1.2
- CUDA==9.2

```bash
  git clone https://github.com/qqqyd/ADNet.git
  cd ADNet/

  conda create --name ADNet -y
  conda activate ADNet
  conda install ipython
  pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
  pip install -r requirement.txt

  cd assets/ops/dcn/
  python setup.py build_ext --inplace
```

## Testing

Prepare the datasets and put them in ```datasets/```.

Download the trained models in [Google Drive](https://drive.google.com/drive/folders/1CBe5qQGJPVAA48BEX-wBX72200PXXK9Z?usp=share_link) and put them in ```models/```.

Evaluate the models using following commands:
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext.yaml --resume models/adnet_td500 --polygon --box_thresh 0.7
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500.yaml --resume models/adnet_td500 --polygon --box_thresh 0.7
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ctw1500.yaml --resume models/adnet_td500 --polygon --box_thresh 0.8
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15.yaml --resume models/adnet_td500 --polygon --box_thresh 0.8
```

## Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py <path-to-yaml-file> --name <task-name> --resume <pretrained model on SynthText (optional)> --num_gpus 4
```



## Citing the related works

If you find our method useful for your reserach, please cite

    @ARTICLE{qu2022adnet,
      author={Qu, Yadong and Xie, Hongtao and Fang, Shancheng and Wang, Yuxin and Zhang, Yongdong},
      journal={IEEE Transactions on Multimedia}, 
      title={ADNet: Rethinking the Shrunk Polygon-Based Approach in Scene Text Detection}, 
      year={2022},
      pages={1-14},
      doi={10.1109/TMM.2022.3216729}}
