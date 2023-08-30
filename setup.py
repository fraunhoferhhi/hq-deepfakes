import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="deepfake_arian",
    py_modules=["deepfake_arian"],
    version="1.2.3",
    description="",
    author="Arian Beckmann",
    packages=find_packages(exclude=["tests*"]),
    install_requires = [
        str(r) 
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True
)