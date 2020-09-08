# Automatic Human removal using Image Segmentation in Deep Learning (dl4j) + Seam Carving Algorithm in Java

## Animated Demo
![Demo](demo/demo.gif)

## Before And After

<img src="demo/request.jpg" width="250"> <img src="demo/response.jpeg" width="250">

## How to run

```
# Building the docker image
docker build . -t remove-human-dl4j
```
```
# Run it as web server
docker run --rm -p 5000:5000 remove-human-dl4j
```

```
curl http://localhost:5000/removehuman --data-binary "@/path_to_image/image.jpg" --output /output_path/output.jpg
```

## How it works

### Image segementaion using deep learning
This project imports a Deep Learning Image segmentation model pretrained on tensorflow (http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz) to Deeplearning4j (nd4j) for detecting human from the photo

Input:
<img src="demo/request.jpg" width="250"> 

Mask:
<img src="demo/mask.jpg" width="250">

### Content-aware image resizing in Computer Vision (Seam Carving Algorithm)
With the masked area detected, we make use of Seam Carving Algorithm (https://en.wikipedia.org/wiki/Seam_carving) with energy in masked area set to zero so that all seams will pass through the mask area during the down sizing. After all masked area removed, we will be (content-aware) upsizing the image again using the same algorithm.
