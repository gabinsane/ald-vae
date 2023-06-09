# ALD-VAE

This is the official code for the submitted ICONIP 2023 submission "Adaptive Compression of the Latent Space in Variational Autoencoders".

---
### **List of contents**

* [Preliminaries](#preliminaries) <br>
* [Setup & Training](#setup-and-training) <br>
* [Evaluation](#evaluation)<br>
* [Training on other datasets](#training-on-other-datasets) <br>
* [License & Acknowledgement](#license)<br>
---
## Preliminaries

This code was tested with:

- Python version 3.8.13
- PyTorch version 1.12.1
- CUDA version 10.2 and 11.6

We recommend to install the conda enviroment as follows:

```
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
conda activate aldvae                 
```

Please note that the framework depends on the [Pytorch Lightning](https://www.pytorchlightning.ai/) framework which manages the model training and evaluation. 


## Download datasets

We provide a preprocessed version of the MNIST, FashionMNIST, SPRITES and EuroSAT datasets. To download them, run:

For MNIST:

```
cd ~/adaptive_vae
wget https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/mnist_svhn.zip   # download mnist_svhn dataset
unzip mnist_svhn.zip -d ./data/
```

For FashionMNIST:

```
cd ~/adaptive_vae
wget https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/fashionmnist.zip   # download fashionmnist dataset
unzip fashionmnist.zip -d ./data/
```

For SPRITES:

```
cd ~/adaptive_vae
wget https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/sprites.zip   # download sprites dataset
unzip sprites.zip -d ./data/
```

For EuroSAT:

```
cd ~/adaptive_vae
wget https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/eurosat.zip   # download eurosat dataset
unzip eurosat.zip -d ./data/
```


## Training
If you have downloaded and extracted data for your chosen dataset, you can launch any of the selected configs in the configs folder. 


```
python main.py --cfg configs/config_mnist.yml
```


## License

This code is published under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).  
