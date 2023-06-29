# Knowledge Distillation

Knowledge distillation is the process through which outputs 
from one neural network are used to train another neural network.
This is usually useful in particular to:
- train a much smaller model
- Modify a trained model (pruning, compresion etc)
- Train without using labels

Check out the seminal paper by [Hinton et al](https://arxiv.org/abs/1503.02531)


## Coding Assignment

### Supervised KD
By this point you should have trained a simple classification model and
have checkpoints under `checkpoints/vanilla`

Now you should copy the contents of `2_pytorch_primer/train.py` to this directory
and edit it to use the original models predictions as labels. You might want to look at:
- `2_pytorch_primer/train.py:38` 
- `2_pytorch_primer/train.py:46` 
- `src/utils/eval.py`


### Unsupervised KD

Read more about the STL10 Dataset https://cs.stanford.edu/~acoates/stl10/
Change the code to train on unlabeled data