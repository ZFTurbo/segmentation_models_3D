# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import segmentation_models_3D as sm


def tst_keras_simple():
    # encoder_weights = 'imagenet'
    encoder_weights = None
    model1 = sm.Unet(
        'resnet18',
        input_shape=(64, 64, 64, 3),
        encoder_weights=encoder_weights
    )
    print(model1.summary())
    model2 = sm.FPN(
        'resnet34',
        input_shape=(64, 64, 64, 3),
        encoder_weights=encoder_weights
    )
    print(model2.summary())
    model3 = sm.Linknet('efficientnetb0', input_shape=(64, 64, 64, 3), encoder_weights=None)
    print(model3.summary())
    model4 = sm.PSPNet('densenet121', input_shape=(96, 96, 96, 3), encoder_weights=encoder_weights)
    print(model4.summary())


def tst_keras_all_models():
    encoder_weights = None
    list_of_models = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50',
        'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101',
        'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2',
        'inceptionresnetv2', 'inceptionv3', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2',
        'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7',
        'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnetv2-s',
        'efficientnetv2-m', 'efficientnetv2-l'
    ]

    for backbone in list_of_models:
        print('Go for backbone: {}'.format(backbone))

        shape_size = (128, 128, 128, 3)

        model1 = sm.Unet(
            backbone,
            input_shape=shape_size,
            encoder_weights=encoder_weights,
            classes=2,
        )
        print(model1.summary())
        model2 = sm.FPN(
            backbone,
            input_shape=shape_size,
            encoder_weights=encoder_weights,
            classes=1,
        )
        print(model2.summary())
        model3 = sm.Linknet(
            backbone,
            input_shape=shape_size,
            encoder_weights=None,
            classes=4,
        )
        print(model3.summary())
        model4 = sm.PSPNet(
            backbone,
            input_shape=(288, 288, 288, 3),
            encoder_weights=encoder_weights,
            classes=3,
        )
        print(model4.summary())


if __name__ == '__main__':
    # tst_keras_simple()
    tst_keras_all_models()
