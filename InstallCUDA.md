ubuntu 18.04
```
sudo apt install ubuntu-drivers-common
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
sudo reboot
```
Adding the following into the bashrc
```
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```
* https://developer.nvidia.com/rdp/cudnn-download
* Download cuDNN v7.5.1 (April 22, 2019), for CUDA 10.0
* cuDNN Developer Library for Ubuntu18.04 (Deb)
```
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.1/prod/10.0_20190418/Ubuntu18_04-x64/libcudnn7_7.5.1.10-1%2Bcuda10.0_amd64.deb
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.1/prod/10.0_20190418/Ubuntu18_04-x64/libcudnn7-dev_7.5.1.10-1%2Bcuda10.0_amd64.deb
sudo  dpkg -i Downloads/libcudnn7_7.5.1.10-1+cuda10.0_amd64.deb
sudo  dpkg -i libcudnn7-dev_7.5.1.10-1+cuda10.0_amd64.deb
```

ubuntu 16.04
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
Adding the following into the bashrc
```
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```
* https://developer.nvidia.com/rdp/cudnn-download
* Download cuDNN v7.5.1 (April 22, 2019), for CUDA 10.0
* cuDNN Developer Library for Ubuntu18.04 (Deb)
```
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.1/prod/10.0_20190418/Ubuntu18_04-x64/libcudnn7_7.5.1.10-1%2Bcuda10.0_amd64.deb
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.1/prod/10.0_20190418/Ubuntu18_04-x64/libcudnn7-dev_7.5.1.10-1%2Bcuda10.0_amd64.deb
sudo  dpkg -i libcudnn8_8.1.1.33-1+cuda10.2_amd64_1604.deb
sudo  dpkg -i libcudnn8-dev_8.1.1.33-1+cuda10.2_amd64_1604.deb
```
