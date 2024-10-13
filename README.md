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
python main.py --cfg configs/mnist_aldvae.yml
```

To switch the latent space compression mechanism on/off, you can change the "adaptive" parameter in the config from 1 to 0 (as is done e.g. in mnist_fixed.yml). You can also change parameters such as initial_latent_n (how many neurons to remove at once in each pruning) and initial_patience (perform pruning every xx epochs). The parameter n_latents is the initial latent dimensionality in ALD-VAE and the fixed dimensionality in the non-adaptive scenario.


## Citation

```
@inproceedings{sejnova2024adaptive,
  title={Adaptive Compression of the Latent Space in Variational Autoencoders},
  author={Sejnova, Gabriela and Vavrecka, Michal and Stepanova, Karla},
  booktitle={International Conference on Artificial Neural Networks},
  pages={89--101},
  year={2024},
  organization={Springer}
}
```
