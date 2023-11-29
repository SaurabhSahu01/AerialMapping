import cv2
import glob
import imageprocessing as imgproc

def processImages(dataset_path, algo):
    # reading all the image paths in a list
    image_paths = glob.glob(f"{dataset_path}/*.jpeg")
    image_paths = sorted(image_paths)
    images = []

    for image in image_paths:
        img = cv2.imread(image)
        images.append(img)

    stitcher = imgproc.ImageStitching()

    if algo == "Cluster":
        stitcher.stitch_images_cluster(images)
    elif algo == "Sequentially":
        stitcher.stitch_images_sequentially(images)
    elif algo == "Random":
        stitcher.stitch_images_random(images)
    else:
        stitcher.stitch_images_greedy(images)
    
    
    # stitcher.generate_all_matches(images, "unstitchedImages")

if __name__ == "__main__":
    processImages('./newdataset/*.jpeg', "ram")
