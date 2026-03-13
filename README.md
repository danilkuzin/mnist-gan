# mnist-gan
Training: 
- *Vanilla dense GAN on MNIST* Minimal GAN example before introducing modern techniques.
- *DCGAN on CelebA*

## Running
Build docker image
```docker
docker build -t mnist-gan .
```
Run docker container
```docker
docker run --privileged --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $PWD:/source -v <path_to_data_folder>:/data mnist-gan
```
Run training
```python
python run_train.py vanilla_mnist
```

## Vanilla GAN - MNIST

### Model
**Generator**
* 3-layer MLP
* ReLU
* sigmoid

**Discriminator**
* 2-layer Maxout
* Dropout
* Final linear projection
* Binary classification

### Data and hyperparameters
* MNIST train
* SGD (momentum=0.5)
* lr d = 1e-1
* lr g = 1e-1
* latent dim = 100
* batch size = 100

### Losses
BCE. 
* ![D fake](results/vanilla_mnist/D_fake.png) 
* ![D real](results/vanilla_mnist/D_real.png) 
* ![G](results/vanilla_mnist/G.png) 

### Generations
Samples generated from a fixed latent vector.
* Initial ![initial](results/vanilla_mnist/samples_iter_0.png)
* 10 epochs ![10 epochs](results/vanilla_mnist/samples_iter_10.png)
* 20 epochs ![20 epochs](results/vanilla_mnist/samples_iter_20.png)
* 30 epochs ![30 epochs](results/vanilla_mnist/samples_iter_30.png)
* 40 epochs ![40 epochs](results/vanilla_mnist/samples_iter_40.png)
* 50 epochs ![50 epochs](results/vanilla_mnist/samples_iter_50.png)

### Interpolations
![](results/vanilla_mnist/interpolated_iter_190.png)

## DCGAN - CelebA

### Model
**Generator**
* Blocks: 4 * ConvTrasnpose/BatchNorm/ReLU
* ConvTrasnpose
* tanh

**Discriminator**
* Blocks: Conv / BatchNorm/ LeakyReLU

### Data and hyperparameters
* CelebA 
* Adam (betas=(0.5, 0.999))
* lr d = 1e-4
* lr g = 3e-4
* latent dim = 100
* batch size = 128

### Generations
* ![](results/dcgan_celeba/generations.png)

### Interpolations
* ![](results/dcgan_celeba/interpolations.png)
