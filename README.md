# Wheat Head Counting by Estimating a Density Map with Convolutional Neural Networks

Wheat is one of the most significant crop species with an annual worldwide grain production of 700
million tonnes. Assessing the production of wheat spikes can help us measure the grain production. Thus, detecting and characterizing spikes from images of wheat fields is an essential component in a wheat breeding process. 

In this study, we propose three wheat head counting networks (WHCNet\_1, WHCNet\_2 and WHCNet\_3) to accurately estimate the wheat head count from an individual image and construct high quality density map, which illustrates the distribution of wheat heads in the image. The WHCNets are composed
of two major components: a convolutional neural network (CNN) as the front-end for wheat head image feature extraction and a CNN with skip connections for the back-end to generate high-quality density maps.  We compare our methods with CSRNet, a deep learning
method which developed for highly congested scenes understanding and 
performing accurate count estimation as well as presenting high
quality density maps. By taking the advantage of the skip connections between CNN layers, WHCNets integrate features  from low CNN layers to high CNN layers, 
thus, the output density maps have both high spatial resolution and detailed representations of the input images. 
The experiments showed that our methods outperformed CSRNet in terms of  the evaluation metrics, mean  absolute  error  (MAE)  and  the  root  mean squared  error  (RMSE) with smaller model sizes. 

# Method
Density map based wheat head counting refers to the input is a wheat head image and the output is the density map of the wheat heads, which shows how many wheat heads per unit area and the spatial distribution of wheat heads in that image, so it is very useful in many applications, such as, estimating the grain yield potential. Consequently, the number of wheat heads in an image can be obtained by the integration of its density map.

First, we will introduce the dataset and data preprocessing, then, we will discuss how to generate the ground truth density maps from wheat head images. Third, we present three wheat head counting networks, WHCNet\_1, WHCNet\_2 and WHCNet\_3, which can learn density maps from input wheat head images via fully CNNs. 

## Dataset and data preprocessing
Global wheat head detection (GWHD) dataset (https://www.kaggle.com/c/global-wheat-detection) is collected from several countries around the world at different growth stages with a wide range of genotypes aiming at developing and benchmarking methods for wheat head detection. 

![plot](./Pic1.png)

Figure 1. shows the distribution of the count number of bounding boxes per image. As can be seen from the figure, most of the images have 20-60 wheat heads, and few images, specifically 4 images, contain more than 100 heads with a maximum of 116 heads. Moreover, there are 49 images containing no heads in the dataset. 

## Ground truth density map generation
![plot](./Pic2.png)

## WHCNet architecture


![plot](./Pic6.png)
![plot](./Pic7.png)
![plot](./Pic8.png)

# Results

![plot](./Pic3.png)
![plot](./Pic4.png)
![plot](./Pic5.png)

# Read this paper for detailed description. 
<a href="https://arxiv.org/abs/2303.10542" class="image fit">Wheat Head Counting by Estimating a Density Map with
Convolutional Neural Networks </a>
