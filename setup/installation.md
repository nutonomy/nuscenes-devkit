## Advanced Installation
We provide step-by-step instructions to install our devkit. 
- [Source Download](#source-download)
- [Install Python](#install-python)
- [Setup a CONDA environment](#setup-a-conda-environment)
- [Setup PYTHONPATH](#setup-pythonpath)
- [Install required packages](#install-required-packages)
- [Verify Install](#verify-install)
- [Setup NUSCENES env variable](#setup-nuscens-env-variable)
- [(Alternative) Setup a virtualenv environment](#(Alternative)-Setup-a-virtualenv-environment)


### Source Download

Download the devkit to your home directory using:
```
cd && git clone https://github.com/nutonomy/nuscenes-devkit.git
```
### Install Python

The devkit is tested for Python 3.6 onwards, but we recommend to use Python 3.7. For Ubuntu: If the right Python version isn't already installed on your system, install it by doing
```
sudo apt install python-pip
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
sudo apt-get install python3.7-dev
```
For Mac OS: download from `https://www.python.org/downloads/mac-osx/` and install.

### Setup a CONDA environment
It is then recommended to install the devkit in a new virtual environment, follow instructions below to setup your dev environment.  Here we include instructions for [CONDA](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you don't want to install conda, an alternative would be to use virtualenvwrapper, if you prefer this, you can look at [these instructions](#alternative-setting-up-a-new-virtual-environment).

##### Install miniconda
https://conda.io/en/latest/miniconda.html

##### Setup a CONDA environment
We create a new vitrual environment named `nuscenes`.
```
conda create --name nuscenes python=3.7
```

##### Activating the virtual environment
If you are inside the virtual environment, your shell prompt should look like: `(nuscenes) user@computer:~$`
If that is not the case, you can enable the virtual environment using:
```
conda activate nuscenes 
```
To deactivate the virtual environment, use:
```
source deactivate
```
### Setup PYTHONPATH
Add the `python-sdk` directory to your `PYTHONPATH` environmental variable, by adding the 
following to your `~/.bashrc` (For virtualenvwrapper, you could add it in `~/.virtualenvs/nuscenes/bin/postactivate`):
```
export PYTHONPATH="${PYTHONPATH}:$HOME/nuscenes-devkit/python-sdk"
```

### Install required packages

To install the required packages, run the following command in your favourite virtual environment:
```
pip install -r setup/requirements.txt
```

### Verify install
To verify your environment run `python -m unittest` in the `python-sdk` folder.
You can also run `assert_download.py` in the `nuscenes/scripts` folder.

### Setup NUSCENES environment variable

Finally, set NUSCENES env. variable that points to your data folder. In the script below, our data is stored in `/data/sets/nuscenes`.
```
export NUSCENES="/data/sets/nuscenes"
```

That's it you should be good to go!

-----
### (Alternative) Setup a virtualenv environment
Another option for setting up a new virtual environment is to use virtualenvwrapper.  Follow instructions below to setup your dev environment.

##### Install virtualenvwrapper
```
pip install virtualenvwrapper
```
Add these two lines to `~/.bashrc` (`~/.bash_profile` on MAC OS) to set the location where the virtual environments 
should live and the location of the script installed with this package:
```
export WORKON_HOME=$HOME/.virtualenvs
source [VIRTUAL_ENV_LOCATION]
```
Replace `[VIRTUAL_ENV_LOCATION]` with either `/usr/local/bin/virtualenvwrapper.sh` or `~/.local/bin/virtualenvwrapper.sh` 
depending on where it is installed on your system.

After editing it, reload the shell startup file by running e.g. `source ~/.bashrc`.

##### Create the virtual environment
We create a new virtual environment named `nuscenes`.
```
mkvirtualenv nuscenes --python=python3.7 
```
or
```
mkvirtualenv nuscenes --python [PYTHON_BINARIES] 
```
PYTHON_BINARIES are typically at either `/usr/local/bin/python3.7` or `/usr/bin/python3.7`.

##### Activating the virtual environment
If you are inside the virtual environment, your shell prompt should look like: `(nuscenes) user@computer:~$`
If that is not the case, you can enable the virtual environment using:
```
workon nuscenes
```
To deactivate the virtual environment, use:
```
deactivate
```