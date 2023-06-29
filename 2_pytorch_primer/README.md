```bash
python 2_pytorch_primer/train.py --epochs 10 --checkpoint checkpoints/vanilla
python eval.py vanilla.pt
jupyter notebook --no-browser -NotebookApp.token='ABC'
```

### Pytorch Primer (30 min)

Objectives:
- Know what is pytorch
- Have run and modified a pytorch training script
- Train a neural network and observe the output
- Understand what is complexity in neural networks

### Model Complexity
Many things can be used to describe complexity in deep neural networks:
number of parameters, size in MB, number of FLOPS, number of activations,
inference latency, training latency, energy consumption etc

We will explore in `2_pytorch_primer/complexity.ipynb` the most common proxy: FLOPS
But if you want to go ahead and look at `4_split_computing/latency.py` we will also explore
inference latency.

Recommended reading: Designing Network Design Spaces - https://arxiv.org/abs/2003.13678




