# Advanced Installation
We provide step-by-step instructions to install our devkit. 
- [Download](#download)
- [Install Python](#install-python)
- [Setup a Conda environment](#setup-a-conda-environment)
- [Setup a virtualenvwrapper environment](#setup-a-virtualenvwrapper-environment)
- [Setup PYTHONPATH](#setup-pythonpath)
- [Install required packages](#install-required-packages)
- [Verify install](#verify-install)
- [Setup NUSCENES environment variable](#setup-nuscenes-environment-variable)

## Download

Download the devkit to your home directory using:
```
cd && git clone https://github.com/nutonomy/nuscenes-devkit.git
```
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
We create a new Conda environment named `nuscenes`.
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

## Verify install
To verify your environment run `python -m unittest` in the `python-sdk` folder.
You can also run `assert_download.py` in the `nuscenes/scripts` folder.

## Setup NUSCENES environment variable
Finally, if you want to run the unit tests you need to point the devkit to the `nuscenes` folder on your disk.
Set the NUSCENES environment variable to point to your data folder, e.g. `/data/sets/nuscenes`:
```
export NUSCENES="/data/sets/nuscenes"
```

That's it you should be good to go!