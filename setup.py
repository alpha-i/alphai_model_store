from setuptools import setup
from setuptools import find_packages


setup(
    name='alphai_model_store',
    version='0.0.4',
    description='Collection of models for multivariate time-series analysis',
    author='Parvez Alam Kazi, William Tai, Gabriele Alese',
    author_email='parvez.alam.kazi@alpha-i.co, william.tai@alpha-i.co, gabriele.alsese@alpha-i.co',
    packages=find_packages(exclude=['docs', 'tests*']),
    install_requires=[
        'pandas==0.22',
        'requests',
        "h5py==2.7.1",
        "tensorflow==1.4.0",
        "scipy",
        "sklearn",
        "contexttimer",
        "tables==3.4.2",
        "matplotlib",
        "pillow",
        "alphai_es_datasource"
    ],
    dependency_links=[
        'https://repo.fury.io/yuTgVkyQQ8f7868U91LP/alpha-i/'
    ]
)
