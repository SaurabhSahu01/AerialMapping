import cv2
import numpy as np
import random
import time
import sys

class ImageStitching:
    def __init__(self):
        pass

    def number_of_matches(self, images: list):
        '''
            This function takes the list of images containing 2 images <[image1, image2]>
            and after calculating the keypoints and descriptors, return the the best matches
            and the homography matrix <(filtered_matches, M)>
        '''
        # Convert the images to grayscale
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        keypoints_and_descriptors = [orb.detectAndCompute(img, None) for img in gray_images]

        # Use BFMatcher to find the best matches between descriptors
        # cv2.BFMatcher with crossCheck=True: This is an extension of the brute-force matcher
        # with cross-checking enabled. It helps filter out false matches by ensuring that the
        # match from the first set to the second set is the same as the match from the second
        # set to the first set.
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(keypoints_and_descriptors[0][1], keypoints_and_descriptors[1][1])

        # Extract the matched keypoints
        src_pts = np.float32([keypoints_and_descriptors[0][0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_and_descriptors[1][0][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Apply the mask to filter out outliers
        filtered_matches = [matches[i] for i in range(len(matches)) if mask[i][0] == 1]

        return filtered_matches, M

    def generate_all_matches(self, images, output_name):
        '''
            input - list of all the images
            output - generates a matrix as output in the file(./observation/output_name.txt)
            matrix - show the number of matches between each pair of images
        '''
        with open(f"./observations/{output_name}.txt", "w") as f:
            sys.stdout = f
            for i in range(len(images)):
                for j in range(len(images)):
                    filtered_matches, M = self.number_of_matches([images[i], images[j]])
                    print(f"{len(filtered_matches)}   ", end="")
                print("\n")

    def best_matched_images(self, images: list) -> tuple:
        '''
            input - list of all the images
            output - return the indices of two images from the passed list having the highest number of best matches
        '''
        max_so_far = -10
        indices = []
        for i in range(len(images)):
                for j in range(len(images)):
                    if(i != j):
                        filtered_matches, M = self.number_of_matches([images[i], images[j]])
                        if(len(filtered_matches) > max_so_far):
                            max_so_far = len(filtered_matches)
                            indices = [i, j]
        print("max match : ", max_so_far)
        return indices
                

    def stitch_2_images(self, images: list):
        '''
            input - a list containing 2 images that are to be stitched
            output - stiched image
        '''
        # Convert the images to grayscale
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]


        filtered_matched, M = self.number_of_matches(images)

        # Get the dimensions of the first image
        h1, w1 = gray_images[0].shape

        # Get the dimensions of the second image
        h2, w2 = gray_images[1].shape

        # Define the corners of the first image
        corners = np.array([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]], dtype=np.float32).reshape(-1, 1, 2)

        # Transform the corners using the homography matrix
        transformed_corners = cv2.perspectiveTransform(corners, M)

        # Calculate the dimensions of the stitched image
        min_x = int(min(np.min(transformed_corners[:, :, 0]), 0))
        min_y = int(min(np.min(transformed_corners[:, :, 1]), 0))
        max_x = int(max(np.max(transformed_corners[:, :, 0]), w2 - 1))
        max_y = int(max(np.max(transformed_corners[:, :, 1]), h2 - 1))

        # Adjust the translation part of the homography matrix
        translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        adjusted_homography = np.dot(translation_matrix, M)

        # Warp the first image
        result = cv2.warpPerspective(images[0], adjusted_homography, (int(max_x - min_x + 1), int(max_y - min_y + 1)))

        # Copy the second image onto the result image
        result[-min_y:h2 - min_y, -min_x:w2 - min_x] = images[1]

        return result

    def stitch_images_cluster(self, images: list):
        '''
            input - a list containing all the images that are to be stitched

            algorithm : 
                step 1.) start with the 0th index image
                step 2.) get a list of matches with the first image
                step 3.) select the image with highest number of matches
                step 4.) stitch both the images
                step 5.) put the stitched image in the 0th index and remove the older images which were stitched
                step 6.) repeat step 1 to 5 till we are left with only 1 image in the list
        '''
        current_time = time.time()
        with open(f"./output/cluster/{current_time}.txt", 'w') as f:
            sys.stdout = f

            while(len(images) > 1):
                print("length of images list is : ", len(images))
                matches = []
                starting_index = 0
                print("starting_image is : ", starting_index+1)
                for i in range(len(images)):
                    if(i is not starting_index):
                        filtered_matches, M = self.number_of_matches([images[starting_index], images[i]])
                        matches.append(len(filtered_matches))
                    else:
                        matches.append(0)
                highest_feature_matching_score = max(matches)
                print("highest_feature_matching_score : ", highest_feature_matching_score)
                required_index = matches.index(highest_feature_matching_score)
                print("Required_image : ", required_index+1)

                result = self.stitch_2_images([images[starting_index], images[required_index]])

                cv2.imshow(f"stiched image", result)
                cv2.waitKey(150)

                max_index = max(starting_index, required_index)
                min_index = min(starting_index, required_index)
                images.pop(max_index)
                images.pop(min_index)
                images.insert(0, result)

        # cv2.imshow("final stitched image : ", images[0])
        # cv2.waitKey(0)
        cv2.imwrite(f"./output/cluster/final.jpeg", images[0])

    def stitch_images_random(self, images: list):
        '''
            input - a list containing all the images that are to be stitched

            algorithm: 
                step 1.) choose a random image from the list
                step 2.) get a list of matches with rest of the images in the list
                step 3.) stitch the image with the best matched image
                step 4.) put the stitched image at the back of the list and remove the older images from the list which were stitched
                step 5.) repeat step 1 to 4 until we are left with single image in the list
        '''
        current_time = time.time()
        with open(f"./output/random/{current_time}.txt", 'w') as f:
            sys.stdout = f
        
            while(len(images) > 1):
                print("length of images list is : ", len(images))
                matches = []
                starting_index = random.randint(0, len(images)-1)
                #starting_index = len(images)-1
                # starting_index = 0
                print("starting_image is : ", starting_index+1)
                for i in range(len(images)):
                    if(i is not starting_index):
                        filtered_matches, M = self.number_of_matches([images[starting_index], images[i]])
                        matches.append(len(filtered_matches))
                    else:
                        matches.append(0)
                highest_feature_matching_score = max(matches)
                print("highest_feature_matching_score : ", highest_feature_matching_score)
                required_index = matches.index(highest_feature_matching_score)
                print("Required_image : ", required_index+1)

                result = self.stitch_2_images([images[starting_index], images[required_index]])

                cv2.imshow(f"stiched image", result)
                cv2.waitKey(150)

                max_index = max(starting_index, required_index)
                min_index = min(starting_index, required_index)
                images.pop(max_index)
                images.pop(min_index)
                images.append(result)
                # images.insert(0, result)
        # cv2.imshow("final stitched image : ", images[0])
        # cv2.waitKey(0)
        cv2.imwrite(f"./output/random/final.jpeg", images[0])

    def stitch_images_greedy(self, images: list):
        '''
            input - a list containing all the images to be stitched

            algorithm:
                step 1.) find a pair of best matched image
                step 2.) stich those images
                step 3.) put the stitched image back in the list and remove the older participating images
                step 4.) repeat step 1 to 3 until we are left with a single image in the list
        '''
        current_time = time.time()
        with open(f"./output/greedy/{current_time}.txt", 'w') as f:
            sys.stdout = f

            while(len(images) > 1):
                print("length of images list is : ", len(images))
                indices = self.best_matched_images(images)
                print(f"indices - {indices[0]+1} {indices[1]+1}")
                result = self.stitch_2_images([images[indices[0]], images[indices[1]]])

                cv2.imshow(f"stiched image", result)
                cv2.waitKey(150)

                max_index = max(indices[0], indices[1])
                min_index = min(indices[0], indices[1])

                images.pop(max_index)
                images.pop(min_index)
                images.append(result)
        # cv2.imshow("final stitched image : ", images[0])
        # cv2.waitKey(0)
        cv2.imwrite(f"./output/greedy/final.jpeg", images[0])

    def stitch_images_sequentially(self, images: list):
        '''
            input - a list of images to be stitched

            algorithm:
                step 1.) make sure that the images are in order
                step 2.) stitch the first two images
                step 3.) put the stitched image in the front of the list, and remove the older participants
                step 4.) repeat step 2 to 3 until we are left with single image in the list

            Note : This method is best suited if the images have time stamps or any mark to represent the order of the images.
        '''
        current_time = time.time()
        with open(f"./output/sequentially/{current_time}.txt", 'w') as f:
            sys.stdout = f

            while(len(images) > 1):
                print("length of images list is : ", len(images))
                starting_index = 0
                required_index = 1
                print(f"indices - {starting_index+1} {required_index+1}")
                result = self.stitch_2_images([images[starting_index], images[required_index]])

                cv2.imshow(f"stiched image", result)
                cv2.waitKey(150)

                images.pop(required_index)
                images.pop(starting_index)
                images.insert(0, result)
        # cv2.imshow("final stitched image : ", images[0])
        # cv2.waitKey(0)
        cv2.imwrite(f"./output/sequentially/final.jpeg", images[0])





