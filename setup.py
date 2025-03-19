import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="hq_deepfakes",
    py_modules=["hq_deepfakes"],
    version="1.0.0",
    description="",
    author="Arian Beckmann",
    packages=find_packages(exclude=["tests*"]),
    install_requires = [
        str(r) 
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    package_data={
        "hq_deepfakes": ["conversion/pretrained/799999_iter.pth"],
    }
)