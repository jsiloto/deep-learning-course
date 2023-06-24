```bash
python src/train.py --epochs 1 --output vanilla.pt
python eval.py vanilla.pt
jupyter notebook --no-browser -NotebookApp.token='ABC'
```

### Pytorch Primer (30 min)

Main libraries -> We are focusing on PyTorch

By today you should:
- Know how to use pytorch
	- Get Script for pytorch loop
	- Add new model
	- Fix bug with topk
	- Train a model (Give weights)
	- Observe Output



### Model Complexity
Designing Network Design Spaces - https://arxiv.org/abs/2003.13678
Number of Activations


```python
import ptflops
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = mobilenet_small()
  net = mobilenet_large()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
```

