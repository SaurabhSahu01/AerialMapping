import cv2
import glob
import imageprocessing as imgproc

# reading all the image paths in a list
image_paths = glob.glob('./newdataset/*.jpeg')
image_paths = sorted(image_paths)
images = []

for image in image_paths:
    img = cv2.imread(image)
    images.append(img)

stitcher = imgproc.ImageStitching()

# stitcher.stitch_images_randomly(images)
# stitcher.stitch_images_greedy(images)
# stitcher.stitch_images_sequentially(images)
stitcher.stitch_images_cluster(images)
# stitcher.generate_all_matches(images, "unstitchedImages")
