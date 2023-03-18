# Adjustable Robust Transformer for High Myopia Screening in Optical Coherence Tomography

Adjustable Robust Transformer (ARTran) is a method for high myopia screening. It can change the preference of the inference result according to the provided adjustment
coefficient. 

<img src="figs/fig-2.png">


## Training
In the training stage, the inclusion threshold is varied around the benchmark, affecting the supervision consequently. The ACE adaptively changes the input state according to the scale of the adjustment coefficient to obtain the corresponding output.
```
# An example for training on 2 GPUs:
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --fold=1 --epoch=100 --batch-size=128 --num-classes=2 --hw-shape=224,224 --kernel-size=56,8 --stride=28,8 --dataroot='' --output-dir=''
```


## Inferencing
In the testing stage, the ACE interactively makes the model output different results depending on the given conditions.
```
# An example for inferencing:
python test.py --shift=0. --fold=1 --batch-size=100 --num-classes=2 --hw-shape=(224, 224) --kernel-size=(56, 8) --stride=(28, 8) --dataroot='' --resume=''
```


## Citing
- TO DO
