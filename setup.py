try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='segmentation_models_3D',
    version='1.0.7',
    author='Roman Sol (ZFTurbo)',
    packages=['segmentation_models_3D', 'segmentation_models_3D/backbones', 'segmentation_models_3D/base', 'segmentation_models_3D/models'],
    url='https://github.com/ZFTurbo/segmentation_models_3D',
    description='Set of Keras models for segmentation of 3D volumes .',
    long_description='3D variants of popular models for segmentation like FPN, Unet, Linknet and PSPNet.'
                     'Models work with keras and tensorflow.keras.'
                     'More details: https://github.com/ZFTurbo/segmentation_models_3D',
    install_requires=[
        'keras>=3.0.0',
        "classification_models_3D==1.0.10",
    ],
)
