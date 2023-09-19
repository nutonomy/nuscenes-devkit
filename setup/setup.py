import os

import setuptools

with open('../README.md', 'r') as fh:
    long_description = fh.read()

# Since nuScenes 2.0 the requirements are stored in separate files.
with open('requirements.txt') as f:
    req_paths = f.read().splitlines()
requirements = []
for req_path in req_paths:
    if req_path.startswith('#'):
        continue
    req_path = req_path.replace('-r ', '')
    with open(req_path) as f:
        requirements += f.read().splitlines()


def get_dirlist(_rootdir):
    dirlist = []

    with os.scandir(_rootdir) as rit:
        for entry in rit:
            if not entry.name.startswith('.') and entry.is_dir():
                dirlist.append(entry.path)
                dirlist += get_dirlist(entry.path)

    return dirlist


# Get subfolders recursively
os.chdir('..')
rootdir = 'python-sdk'
packages = [d.replace('/', '.').replace('{}.'.format(rootdir), '') for d in get_dirlist(rootdir)]

# Filter out Python cache folders
packages = [p for p in packages if not p.endswith('__pycache__')]

setuptools.setup(
    name='nuscenes-devkit',
    version='1.1.11',
    author='Holger Caesar, Oscar Beijbom, Qiang Xu, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, '
           'Sergi Widjaja, Kiwoo Shin, Caglayan Dicle, Freddy Boulton, Whye Kit Fong, Asha Asvathaman, Lubing Zhou '
           'et al.',
    author_email='nuscenes@motional.com',
    description='The official devkit of the nuScenes dataset (www.nuscenes.org).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nutonomy/nuscenes-devkit',
    python_requires='>=3.6',
    install_requires=requirements,
    packages=packages,
    package_dir={'': 'python-sdk'},
    package_data={'': ['*.json']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        'License :: Free for non-commercial use'
    ],
    license='apache-2.0'
)
