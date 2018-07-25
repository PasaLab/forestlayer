# -*- coding:utf-8 -*-
"""
Setup configure.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'scipy',
    'scikit-learn',
    'keras',
    'ray',
    'joblib',
    'xgboost',
    'psutil',
    'matplotlib',
    'pandas'
]

setup(
    name="forestlayer",
    version="0.1.6",
    include_package_data=True,
    author="ForestLayer Contributors",
    author_email="huqiu00@163.com",
    url="https://github.com/whatbeg/forestlayer",
    license="apache",
    packages=find_packages(),
    install_requires=install_requires,
    description="A scalable and fast deep forest learning library.",
    keywords="deep learning, deep forest, machine learning, random forest, ray, keras, xgboost",
    platforms=['any'],
)
