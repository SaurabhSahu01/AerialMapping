# Aerial Mapping

## Technology Used
- Computer Vision
- Python
- tkinter for GUI
- opencv(cv2) for image processing
- Git and GitHub
- Python Libraries Used - cv2, numpy, random, time, sys, glob

<span style="color: skyblue; font-weight: 20px;">BRIEF OVERVIEW OF THE METHODS USED:</span>

```stitch_images_sequentially()``` - start with the first image, stitch images order wise(second, third, fourth, etc) 

```stitch_images_cluster()``` - starting with the first index, find the best match at each iteration to stitch together. Put the stitched image at the first index(0). 

```stitch_images_random()``` - at each iteration, a random point is chose and it's corresponding best matched image is chosen to stitch together. 

```stitch_images_greedy()``` - at each iteration choose the best matching images to stitch together

```Python
   # How to use the methods

    import imageprocessing

    # read the images in the images list
    images = []

    # initialise an instance for the ImageStitching class
    stitcher = imageprocessing.ImageStitching()

    # use the desired method to stitch the images
    stitcher.stitch_images_random(images)
```

<span style="color: orange; font-weight: 20px;">REQUIREMENTS:</span>
1. observations folder must exist beforhand in the root directory
2. output folder must exist in the root directory with the following structure 
   
        output ---  
                  | 
                  |-- cluster 
                  |-- greedy 
                  |-- random 
                  |-- sequentially 

3. dataset folder might vary according to the need of the data in the program
4. images in the dataset should(not necessarily) be ordered by time-stamp

<span style="color: red; font-weight: 20px;">ISSUES:</span>

1. dataset should have data with time stamp to run properly in ```stitch_images_sequentially()```
2. GUI is remaining
