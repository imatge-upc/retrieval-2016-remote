# retrieval-2016-remote

# Multi-Label Remote Sensing Image Retrieval by Using Deep Fetures

| ![Michele Compri][MicheleCompri-photo] | ![Begum Demir][BegumDemir-photo] | ![Xavier Giro-i-Nieto][XavierGiro-photo]| ![Noel O'Connor][NoelOConnor-photo] |
|:-:|:-:|:-:|:-:|
| [Marc Assens][MicheleCompri-web]  | [Begum Demir][BegumDemir-web]  | [Xavier Giro-i-Nieto][XavierGiro-web] |


[MicheleCompri-web]: https://www.linkedin.com/in/marc-assens-reina-5b1090bb/
[BegumDemir-web]: http://begumdemir.com/
[XavierGiro-web]: https://imatge.upc.edu/web/people/xavier-giro

[MicheleCompri-photo]: https://github.com/massens/saliency-360salient-2017/raw/master/authors/foto_carnet_dublin.jpg "Marc Assens"
[BegumDemir-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/authors/Kevin160x160%202.jpg?token=AFOjyZmLlX3ZgpkNe60Vn3ruTsq01rD9ks5YdAaiwA%3D%3D "Begum Demir"
[XavierGiro-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/XavierGiro.jpg "Xavier Giro-i-Nieto"

A joint collaboration between:

| ![logo-trento] | ![logo-gpi] |
|:-:|:-:|:-:|
| [Insight Centre for Data Analytics][insight-web] | [Dublin City University (DCU)][dcu-web] | [UPC Image Processing Group][gpi-web] |

[trento-web]: https://www.insight-centre.org/ 
[gpi-web]: https://imatge.upc.edu/web/ 


[logo-trento]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-gpi]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/gpi.png "UPC Image Processing Group"


## Abstract

Recent advances in satellite technology has led to an increased volume of remote sensing (RS) image archives, from which retrieving useful information is challenging. Therefore, one important research area in remote sensing (RS) is the content-based retrieval of RS images (CBIR). The performance of the CBIR systems relies on the capability of the RS image features in modeling the content of the images as well as the considered retrieval algorithm that assesses the similarity among the features. Using supervised classification methods in the context of CBIR by training the classifier with the already annotated images has attracted attention in RS. However, existing supervised CBIR systems in the RS literature assume that each training image is categorized by only a single label that is associated to the most significant content of the image. However, RS images usually have complex content, i.e., there are usually several regions within each image related to multiple land-cover classes. Thus, available supervised CBIR systems are not capable of accurately characterizing and exploiting the high level semantic content of RS images for retrieval problems.
To overcome these problems and to effectively characterize the high-level semantic content of RS images in supervised CBIR problems, we investigate effectiveness of different deep learning architectures in the framework of multi-label remote sensing image retrieval. It is worth noting that deep learning architectures such as CNNs have recently attracted great attention in RS [1,2] due to its effective and accurate feature learning. However, according to our knowledge this is the first work that deals with adaptation of CNN models to multi-label RS image retrieval problems. This is achieved based on a two-steps strategy. In the first step, a Convolutional Neural Network (CNN) pre-trained for image classification with the ImageNet dataset is used off-the-shelf as a feature extractor. In particular, three popular architectures are explored: 1) VGG16; 2) Inception V3; and 3) ResNet50. VGG16 is a CNN characterized by 16 convolutional layers of stacked 3x3 filters, with intermediate max pooling layers and 3 fully connected layers at the end. Inception V3 is an improved version of the former GoogleNet, which contains more layers but less parameters, by removing fully connected layers and using a global average pooling from the last convolutional layer. ResNet50 is even deeper thanks to the introduction of residual layers, that allow data to flow by skipping the convolutional blocks. In the second step of our research, we modify these three off-the-shelf models by fine-tuning their parameters with a subset of RS images and their multi-label information. Experiments carried out on an archive of aerial images show that fine-tuning CNN architectures with annotated images with multi-labels significantly improve the retrieval accuracy with respect to the standard CBIR methods. We find that fine-tuning using with a multi-class approach achieves better results than considering each label as an independent class. 

## Slides

<iframe src="//www.slideshare.net/slideshow/embed_code/key/aur7h9ST7R35Oa" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/xavigiro/multilabel-remote-sensing-image-retrieval-based-on-deep-features" title="Multi-label Remote Sensing Image Retrieval based on Deep Features" target="_blank">Multi-label Remote Sensing Image Retrieval based on Deep Features</a> </strong> from <strong><a target="_blank" href="https://www.slideshare.net/xavigiro">Xavier Giro</a></strong> </div>


## Software frameworks: Keras

The model is implemented in [Keras](https://github.com/fchollet/keras/tree/master/keras), which at its time is developed over [Theano](http://deeplearning.net/software/theano/).
```
pip install -r https://github.com/massens/saliency-360salient-2017/blob/master/requirements.txt
```

## Acknowledgements

We would like to especially thank Albert Gil Moreno from our technical support team at the Image Processing Group at the UPC, as well as Albert Jimenez for his support with Keras.

| ![AlbertGil-photo]  |
|:-:|
| [Albert Gil](AlbertGil-web)   |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeoForce GTX [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |
|  This work has been developed in the framework of the projects [BigGraph TEC2013-43935-R](https://imatge.upc.edu/web/projects/biggraph-heterogeneous-information-and-graph-signal-processing-big-data-era-application) and [Malegra TEC2016-75976-R](https://imatge.upc.edu/web/projects/malegra-multimodal-signal-processing-and-machine-learning-graphs), funded by the Spanish Ministerio de Economía y Competitividad and the European Regional Development Fund (ERDF).  | ![logo-spain] | 

[logo-nvidia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"
[logo-spain]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/MEyC.png "Logo of Spanish government"

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/imatge-upc/retrieval-2016-remote/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:xavier.giro@upc.edu>.
