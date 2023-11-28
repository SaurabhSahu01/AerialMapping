ISSUES:

1. dataset should have data with time stamp to run properly in stitch_images_sequentially()
2. GUI is remaining


BRIEF Overview of Algorithms Used :

stitch_images_sequentially() - start with the first image, stitch images order wise(second, third, fourth, etc)
stitch_images_cluster() - starting with the first index, find the best match at each iteration to stitch together. Put the stitched image at the first index(0).
stitch_images_random() - at each iteration, a random point is chose and it's corresponding best matched image is chosen to stitch together.
stitch_images_greedy() - at each iteration choose the best matching images to stitch together