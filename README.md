# Futuristic Car Prototype Image Generation

## Overview
Our approach is based on the idea that upcoming new car designs will be similar to existing car designs in their basic structures so that designs don't lose their feasibility of implementation.

Firstly we have trained a **SN-GAN** on a simple car dataset with differential augmentation and LeCam regularizer. This is followed by **fine-tuning** of the trained GAN on a futuristic cars dataset using **FreezeD technique**. Further we have used Res-Net18 to extract latent codes of common and futuristic cars which are then combined using weighted average and fed into SN-GAN.

Weighted average gives us the flexibility to decide the intensity of futuristic flavor which can be added to existing designs to obtain novel ones.

![Pipeline](https://github.com/vivekagarwal2349/KTJ-Datathon/blob/main/pipeline.png)


## Dataset Used
We use 2 datasets to train our models:

S**imple Car Image Dataset:** This dataset contains around 11,000 images of existing cars with a simple design, all the cars are facing left, front-left or front-right. This dataset was a subset of another dataset that was scraped from The Car Connection website and then further classified according to their view. The code to obtain this dataset was taken from following github repository:
https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper

**Futuristic Concept Car Image Dataset:** This dataset contains around 2000 images of concept cars with a futuristic design. All the images in this dataset were scraped from Google Images search results. The code for scraping was written completely by us. 

## SN-GAN
For steady and diverse image generation, we needed a GAN which could effectively absorb the design features of existing car models and successfully reproduce them in subsequent generations. Popular GANS architectures like DC-GAN are plagued by their training instability and inaccurate density ratio estimation by the discriminators. Therefore for our-case we used SN-GAN, a type of General Adversarial Network which uses a weight normalization method called **spectral normalization to stabilize the training of discriminator networks**. In image generation tasks, **it has been verified that generated examples using SN-GAN are more diverse than the conventional weight normalization and achieve better or comparative inception scores relative to previous studies.**

We used StudioGAN 0.2.0 release to train the SNGAN model.  
StudioGAN Repo: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN   

## FreezeD Technique
FeezeD stands for “Freeze the Discriminator” which is a simple baseline for fine-tuning GANs. It's simply freezing the lower layers of the discriminator and only fine-tuning the upper layers performs surprisingly well. FreezeD splits the discriminator into a feature extractor and a classifier and then fine-tunes the classifier only. It’s proven to significantly outperform previous techniques used in GANs. The pre-trained part is frozen and only last layers are trained and how big the change is on the weights in a layer is governed by the learning rate.


