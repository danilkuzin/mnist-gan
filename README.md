# mnist-gan
Training a vanilla dense GAN on MNIST. 

Minimal GAN example before introducing modern techniques.

## Model
**Generator**
* 3-layer MLP
* LeakyReLU
* BatchNorm
* tanh

**Discriminator**
* 3-layer MLP
* LeakyReLU
* Binary classification

## Data and hyperparameters
* MNIST train
* Adam (beta=(0.5, 0.999))
* lr d = 1e-4
* lr g = 3e-4
* latent dim = 64
* batch size = 32

## Losses
BCE. GAN losses are not expected to decrease monotonically.
* ![D fake](results/D_fake.png) 
* ![D real](results/D_real.png) 
* ![G](results/G.png) 

## Generations
Samples generated from a fixed latent vector.
* ![initial](results/samples_iter_-1.png)
* ![9 epoch](results/samples_iter_9.png)
* ![19 epoch](results/samples_iter_19.png)
* ![29 epoch](results/samples_iter_29.png)
* ![39 epoch](results/samples_iter_39.png)
* ![49 epoch](results/samples_iter_49.png)

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
python run_train.py
```
