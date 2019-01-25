import setuptools
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
print(requirements)

setuptools.setup(
    name='nuscenes-devkit',
    version='0.1.9',
    author="Holger Caesar, Qiang Xu, Oscar Beijbom et al.",
    author_email="nuscenes@nutonomy.com",
    description="The official devkit of the nuScenes dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nutonomy/nuscenes-devkit",
    python_requires='>=3.7',
    install_requires=requirements,
    packages=['nuscenes', 'nuscenes.eval', 'nuscenes.export', 'nuscenes.utils'],
    package_dir={'': 'python-sdk'},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)
