# Deep Learning Course

This material is designed for a lab course on computer vision deep learning with advanced topics

Each Section has its own instructions but most source files are shared between modules and located under `src`


## Environment
This material was tested under UBUNTU 20.04/Python3.8.10
it is expected to work on any environment newer than that. To check your environment run
```bash
lsb_release -a
> Distributor ID:	Ubuntu
> Description:	Ubuntu 20.04.5 LTS
> Release:	20.04
> Codename:	focal

python3 -V
> Python 3.8.10
```

Please start by installing the appropriate python environment
```bash
sudo apt-get update
sudo apt-get install python3-pip virtualenv meld git
git clone https://github.com/jsiloto/deep-learning-course.git
cd deep-learning-course/
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

To pre-download the datasets that will be used run python and

```python
from torchvision.datasets.stl10 import STL10
train_dataset = STL10(root="./data/stl10", download=True, split="train")
```



