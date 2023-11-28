import cv2
import numpy as np
import random
import time
import sys

def number_of_matches(images: list) -> int:
    # Convert the images to grayscale
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    keypoints_and_descriptors = [orb.detectAndCompute(img, None) for img in gray_images]

    # Use BFMatcher to find the best matches between descriptors
    '''
    cv2.BFMatcher with crossCheck=True: This is an extension of the brute-force matcher with cross-checking enabled. 
    It helps filter out false matches by ensuring that the match from the first set to the second set is the same 
    as the match from the second set to the first set.
    '''
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

def generate_all_matches(images, output_name):
    with open(f"./observations/{output_name}.txt", "w") as f:
        sys.stdout = f
        for i in range(len(images)):
            for j in range(len(images)):
                filtered_matches, M = number_of_matches([images[i], images[j]])
                print(f"{len(filtered_matches)}   ", end="")
            print("\n")

def best_matched_images(images: list):
    max_so_far = -10
    indices = []
    for i in range(len(images)):
            for j in range(len(images)):
                if(i != j):
                    filtered_matches, M = number_of_matches([images[i], images[j]])
                    if(len(filtered_matches) > max_so_far):
                        max_so_far = len(filtered_matches)
                        indices = [i, j]
    print("max match : ", max_so_far)
    return indices
                

def stitch_2_images(images: list):
    # Convert the images to grayscale
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]


    filtered_matched, M = number_of_matches(images)

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

def stitch_images_cluster(images: list):
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
                    filtered_matches, M = number_of_matches([images[starting_index], images[i]])
                    matches.append(len(filtered_matches))
                else:
                    matches.append(0)
            highest_feature_matching_score = max(matches)
            print("highest_feature_matching_score : ", highest_feature_matching_score)
            required_index = matches.index(highest_feature_matching_score)
            print("Required_image : ", required_index+1)

            result = stitch_2_images([images[starting_index], images[required_index]])

            # cv2.imshow(f"stiched image", result)
            # cv2.waitKey(0)

            max_index = max(starting_index, required_index)
            min_index = min(starting_index, required_index)
            images.pop(max_index)
            images.pop(min_index)
            images.insert(0, result)

    # cv2.imshow("final stitched image : ", images[0])
    # cv2.waitKey(0)
    cv2.imwrite(f"./output/cluster/{current_time}.jpeg", images[0])

def stitch_images_random(images: list):
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
                    filtered_matches, M = number_of_matches([images[starting_index], images[i]])
                    matches.append(len(filtered_matches))
                else:
                    matches.append(0)
            highest_feature_matching_score = max(matches)
            print("highest_feature_matching_score : ", highest_feature_matching_score)
            required_index = matches.index(highest_feature_matching_score)
            print("Required_image : ", required_index+1)

            result = stitch_2_images([images[starting_index], images[required_index]])

            # cv2.imshow(f"stiched image", result)
            # cv2.waitKey(0)

            max_index = max(starting_index, required_index)
            min_index = min(starting_index, required_index)
            images.pop(max_index)
            images.pop(min_index)
            images.append(result)
            # images.insert(0, result)
    # cv2.imshow("final stitched image : ", images[0])
    # cv2.waitKey(0)
    cv2.imwrite(f"./output/random/{current_time}.jpeg", images[0])

def stitch_images_greedy(images: list):
    current_time = time.time()
    with open(f"./output/greedy/{current_time}.txt", 'w') as f:
        sys.stdout = f

        while(len(images) > 1):
            print("length of images list is : ", len(images))
            indices = best_matched_images(images)
            print(f"indices - {indices[0]+1} {indices[1]+1}")
            result = stitch_2_images([images[indices[0]], images[indices[1]]])

            # cv2.imshow(f"stiched image", result)
            # cv2.waitKey(0)

            max_index = max(indices[0], indices[1])
            min_index = min(indices[0], indices[1])

            images.pop(max_index)
            images.pop(min_index)
            images.append(result)
    # cv2.imshow("final stitched image : ", images[0])
    # cv2.waitKey(0)
    cv2.imwrite(f"./output/greedy/{current_time}.jpeg", images[0])

def stitch_images_sequentially(images: list):
    current_time = time.time()
    with open(f"./output/sequentially/{current_time}.txt", 'w') as f:
        sys.stdout = f

        while(len(images) > 1):
            print("length of images list is : ", len(images))
            starting_index = 0
            required_index = 1
            print(f"indices - {starting_index+1} {required_index+1}")
            result = stitch_2_images([images[starting_index], images[required_index]])

            # cv2.imshow(f"stiched image", result)
            # cv2.waitKey(0)

            images.pop(required_index)
            images.pop(starting_index)
            images.insert(0, result)
    # cv2.imshow("final stitched image : ", images[0])
    # cv2.waitKey(0)
    cv2.imwrite(f"./output/sequentially/{current_time}.jpeg", images[0])





