import cv2
import glob
import imageprocessing as imgproc
from datetime import datetime


# reading all the image paths in a list
image_paths = glob.glob('./newdataset/*.jpeg')
image_paths = sorted(image_paths)
images = []

for image in image_paths:
    img = cv2.imread(image)
    images.append(img)

# imgproc.stitch_images_randomly(images)
# imgproc.stitch_images_greedy(images)
# imgproc.stitch_images_sequentially(images)
imgproc.stitch_images_cluster(images)
# imgproc.generate_all_matches(images, "unstitchedImages")
