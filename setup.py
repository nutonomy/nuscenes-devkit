import setuptools
import re
import os
from glob import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


def get_dirlist(rootdir):
    dirlist = []

    with os.scandir(rootdir) as rit:
        for entry in rit:
            if not entry.name.startswith('.') and entry.is_dir():
                dirlist.append(entry.path)
                dirlist += get_dirlist(entry.path)

    return dirlist

# Get subfolders recursively
packages = []
rootdir = 'python-sdk/nuscenes'
packages = [d.replace('/', '.').replace('python-sdk.', '') for d in get_dirlist(rootdir)]

print(packages)

setuptools.setup(
    name='nuscenes-devkit',
    version='0.2.10',
    author="Holger Caesar, Qiang Xu, Oscar Beijbom et al.",
    author_email="nuscenes@nutonomy.com",
    description="The official devkit of the nuScenes dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nutonomy/nuscenes-devkit",
    python_requires='>=3.7',
    install_requires=requirements,
    packages=packages,
    package_dir={'': 'python-sdk'},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],

)
