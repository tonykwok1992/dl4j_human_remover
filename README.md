# Automatic Human removal using Image Segmentation in Deep Leanring (dl4j) + Seam Carving Algorithm in Java

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
