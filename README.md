# Segmentation models 3D Zoo - Keras and TF.Keras

The repository contains 3D variants of popular models for segmentation like FPN, Unet, Linknet and PSPNet. 

This repository is based on great [segmentation_models](https://github.com/qubvel/segmentation_models) repo by [@qubvel](https://github.com/qubvel/)

### Available architectures: 
-  [Unet](https://arxiv.org/abs/1505.04597>)
-  [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
-  [Linknet](https://arxiv.org/abs/1707.03718)
-  [PSPNet](https://arxiv.org/abs/1612.01105)

### Requirements

* tensorflow>=2.3.2
* keras_applications>=1.0.8
* [classification_models_3D](https://github.com/ZFTurbo/classification_models_3D)>=1.0.3
* [efficientnet_3D](https://github.com/ZFTurbo/efficientnet_3D)>=1.0.2

**Note 1**: it's better not to install independent keras module.

**Note 2**: If you have an error try to add the following line in your code `os.environ["KERAS_BACKEND"] = "tensorflow"`

### Installation

`pip install segmentation-models-3D`

### Examples 

##### Loading model:

```python
import segmentation_models_3D as sm

model1 = sm.Unet('resnet34', encoder_weights='imagenet')
# binary segmentation (this parameters are default when you call Unet('resnet34')
model2 = sm.FPN('densenet121', classes=1, activation='sigmoid')
# multiclass segmentation with non overlapping class masks (your classes + background)
model3 = sm.Linknet('resnet34', classes=3, activation='softmax')
# multiclass segmentation with independent overlapping/non-overlapping class masks
model4 = sm.PSPNet('resnet34', classes=3, activation='sigmoid')
# If you need to specify non-standard input shape
model5 = sm.Unet('resnet50', input_shape=(96, 128, 128, 6), encoder_weights=None)
```

All possible backbones: `'resnet18, 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2'`
Also available: `'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7'`

### Related repositories

 * [https://github.com/qubvel/classification_models](https://github.com/qubvel/classification_models) - original classification 2D repo
 * [https://github.com/qubvel/segmentation_models](https://github.com/qubvel/segmentation_models) - original segmentation 2D repo
 * [classification_models_3D](https://github.com/ZFTurbo/classification_models_3D) - models for classification in 3D
 * [volumentations](https://github.com/ZFTurbo/volumentations) - 3D augmentations
 
### Unresolved problems

* There is no 'bilinear' interpolation for UpSample3D layer, so it uses Nearest Neighbour upsampling.
 
## Citation

For more details, please refer to the publication: https://doi.org/10.1016/j.compbiomed.2021.105089

If you find this code useful, please cite it as:
```
@article{solovyev20223d,
  title={3D convolutional neural networks for stalled brain capillary detection},
  author={Solovyev, Roman and Kalinin, Alexandr A and Gabruseva, Tatiana},
  journal={Computers in Biology and Medicine},
  volume={141},
  pages={105089},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2021.105089}
}
```
