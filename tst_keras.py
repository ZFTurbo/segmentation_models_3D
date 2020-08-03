# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


def tst_keras():
    # for keras
    import segmentation_models_3D as sm

    encoder_weights = None
    model1 = sm.Unet('resnet18', input_shape=(64, 64, 64, 3), encoder_weights=encoder_weights)
    print(model1.summary())
    model2 = sm.FPN('resnet34', input_shape=(64, 64, 64, 3), encoder_weights=encoder_weights)
    print(model2.summary())
    model3 = sm.Linknet('efficientnetb0', input_shape=(64, 64, 64, 3), encoder_weights=encoder_weights)
    print(model3.summary())
    model4 = sm.PSPNet('densenet121', input_shape=(96, 96, 96, 3), encoder_weights=encoder_weights)
    print(model4.summary())


if __name__ == '__main__':
    tst_keras()
