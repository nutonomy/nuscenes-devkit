# Advanced Installation
We provide step-by-step instructions to install our devkit. These instructions apply to both nuScenes and nuImages.
- [Download](#download)
- [Install Python](#install-python)
- [Setup a Conda environment](#setup-a-conda-environment)
- [Setup a virtualenvwrapper environment](#setup-a-virtualenvwrapper-environment)
- [Setup PYTHONPATH](#setup-pythonpath)
- [Install required packages](#install-required-packages)
- [Setup environment variable](#setup-environment-variable)
- [Setup Matplotlib backend](#setup-matplotlib-backend)
- [Verify install](#verify-install)

## Download

Download the devkit to your home directory using:
```
cd && git clone https://github.com/nutonomy/nuscenes-devkit.git
```
*Note:* The pip package does not support the prediction or tracking code.
For these, you will need to download the source code and [manually install the requirements](#install-required-packages).

## Install Python

The devkit is tested for Python 3.6 onwards, but we recommend to use Python 3.7.
For Ubuntu: If the right Python version is not already installed on your system, install it by running:
```
sudo apt install python-pip
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
sudo apt-get install python3.7-dev
```
For Mac OS download and install from `https://www.python.org/downloads/mac-osx/`.

## Setup a Conda environment
Next we setup a Conda environment.
An alternative to Conda is to use virtualenvwrapper, as described [below](#setup-a-virtualenvwrapper-environment).

#### Install miniconda
See the [official Miniconda page](https://conda.io/en/latest/miniconda.html).

#### Setup a Conda environment
We create a new Conda environment named `nuscenes`. We will use this environment for both nuScenes and nuImages.
```
conda create --name nuscenes python=3.7
```

#### Activate the environment
If you are inside the virtual environment, your shell prompt should look like: `(nuscenes) user@computer:~$`
If that is not the case, you can enable the virtual environment using:
```
conda activate nuscenes 
```
To deactivate the virtual environment, use:
```
source deactivate
```

-----
## Setup a virtualenvwrapper environment
Another option for setting up a new virtual environment is to use virtualenvwrapper.
**Skip these steps if you have already setup a Conda environment**.
Follow these instructions to setup your environment.

#### Install virtualenvwrapper
To install virtualenvwrapper, run:
```
pip install virtualenvwrapper
```
Add the following two lines to `~/.bashrc` (`~/.bash_profile` on MAC OS) to set the location where the virtual environments should live and the location of the script installed with this package:
```
export WORKON_HOME=$HOME/.virtualenvs
source [VIRTUAL_ENV_LOCATION]
```
Replace `[VIRTUAL_ENV_LOCATION]` with either `/usr/local/bin/virtualenvwrapper.sh` or `~/.local/bin/virtualenvwrapper.sh` depending on where it is installed on your system.
After editing it, reload the shell startup file by running e.g. `source ~/.bashrc`.

Note: If you are facing dependency issues with the PIP package, you can also install the devkit as a Conda package.
For more details, see [this issue](https://github.com/nutonomy/nuscenes-devkit/issues/155). 

#### Create the virtual environment
We create a new virtual environment named `nuscenes`.
```
mkvirtualenv nuscenes --python=python3.7 
```

#### Activate the virtual environment
If you are inside the virtual environment, your shell prompt should look like: `(nuscenes) user@computer:~$`
If that is not the case, you can enable the virtual environment using:
```
workon nuscenes
```
To deactivate the virtual environment, use:
```
deactivate
```

## Setup PYTHONPATH
Add the `python-sdk` directory to your `PYTHONPATH` environmental variable, by adding the following to your `~/.bashrc` (for virtualenvwrapper, you could alternatively add it in `~/.virtualenvs/nuscenes/bin/postactivate`):
```
export PYTHONPATH="${PYTHONPATH}:$HOME/nuscenes-devkit/python-sdk"
```

## Install required packages

To install the required packages, run the following command in your favourite virtual environment:
```
pip install -r setup/requirements.txt
```
**Note:** The requirements file is internally divided into base requirements (`base`) and requirements specific to certain products or challenges (`nuimages`, `prediction` and `tracking`).
From December 2020 onwards we are excluding the `prediction` and `tracking` requirements by default to decrease the number of packages that are installed and avoid incompatibility with other pip packages.
If you want to install these requirements nevertheless, please run the following for `prediction` (and analog for `tracking`):
```
pip install -r setup/requirements/requirements_prediction.txt
``` 

## Setup environment variable
Finally, if you want to run the unit tests you need to point the devkit to the `nuscenes` folder on your disk.
Set the NUSCENES environment variable to point to your data folder:
```
export NUSCENES="/data/sets/nuscenes"
```
or for NUIMAGES:
```
export NUIMAGES="/data/sets/nuimages"
```

## Setup Matplotlib backend
When using Matplotlib, it is generally recommended to define the backend used for rendering:
1) Under Ubuntu the default backend `Agg` results in any plot not being rendered by default. This does not apply inside Jupyter notebooks.
2) Under MacOSX a call to `plt.plot()` may fail with the following error (see [here](https://github.com/matplotlib/matplotlib/issues/13414) for more details):
    ```
    libc++abi.dylib: terminating with uncaught exception of type NSException
    ```
To set the backend, add the following to your `~/.matplotlib/matplotlibrc` file, which needs to be created if it does not exist yet: 
```
backend: TKAgg
```

## Verify install
To verify your environment run `python -m unittest` in the `python-sdk` folder.
You can also run `assert_download.py` in the `python-sdk/nuscenes/tests` and `python-sdk/nuimages/tests` folders to verify that all files are in the right place.

That's it you should be good to go!
