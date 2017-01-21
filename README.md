# squeezeNet

squeezeNet is a much smaller convolutional network, with vastly less amount of parameters.
queezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters. 
Additionally, with model compression techniques SqueezeNet can be compress to less than 0.5MB (510x smaller than AlexNet).

# To use squeezeNet
download the squeezenet.py file, then

```python
from squeezenet import squeezenet

model = squeezenet(classes=10, dim_ordering='th')
model.compile(...)
model.fit(...)
```

# TODO
 - [ ] Compression


# References
 - [SqueezeNet Official Github Repo](https://github.com/DeepScale/SqueezeNet)
 - [SqueezeNet Paper](http://arxiv.org/abs/1602.07360)
