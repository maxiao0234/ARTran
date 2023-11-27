# Adjustable Robust Transformer (ARTran)

This is the official repository for "Adjustable Robust Transformer for High Myopia Screening in Optical Coherence Tomography" (MICCAI 2023)

## Introduction
Adjustable Robust Transformer (ARTran) is a method for high myopia screening. It can change the preference of the inference result according to the provided adjustment
coefficient. We aim to design a high myopia screening method which use a unified model to make different decisions based on different inclusion criteria. We design a label noise learning method shifted subspace transition matrix (SST) to constrain the noisy class-posterior preferences. We have established the association between output preference and adjustment direction that a higher SE as inclusion criteria biases network output more towards positive categories.

<img src="figs/fig-2.png">


## Training
In the training stage, the inclusion threshold is varied around the benchmark, affecting the supervision consequently. The model adaptively changes the input state according to the scale of the adjustment coefficient to obtain the corresponding output.
```
# An example for training on 2 GPUs:
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --fold=1 --epoch=100 --batch-size=128 --num-classes=2 --hw-shape=224,224 --kernel-size=56,8 --stride=28,8 --dataroot='' --output-dir=''
```


## Inferencing
During the inferencing phase, the screening results can be predicted interactively for a given condition (shift).
```
# An example for inferencing:
python test.py --shift=0. --fold=1 --batch-size=100 --num-classes=2 --hw-shape=(224, 224) --kernel-size=(56, 8) --stride=(28, 8) --dataroot='' --resume=''
```


## Citing
If this project is help for you, please cite it.
```
@inproceedings{ma2023adjustable,
  title={Adjustable Robust Transformer for High Myopia Screening in Optical Coherence Tomography},
  author={Ma, Xiao and Zhang, Zetian and Ji, Zexuan and Huang, Kun and Su, Na and Yuan, Songtao and Chen, Qiang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={504--514},
  year={2023},
  organization={Springer}
}
```
