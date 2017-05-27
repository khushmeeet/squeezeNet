# SqueezeNet

squeezeNet is a much smaller convolutional network, with vastly less amount of parameters.
queezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters. 
Additionally, with model compression techniques SqueezeNet can be compress to less than 0.5MB (510x smaller than AlexNet).

# To use SqueezeNet
Do `pip install squeezenet`

```python
from squeezenet import squeezenet

input = ...
output = squeezenet.squeeze_net(input, classes=output_classes)
# returns the computed tensor
# specify softmax function, cost and optimizer
```

# TODO
 - Compression
 - Performance


# References
 - [SqueezeNet Official Github Repo](https://github.com/DeepScale/SqueezeNet)
 - [SqueezeNet Paper](http://arxiv.org/abs/1602.07360)
