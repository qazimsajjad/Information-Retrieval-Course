import cv2
import os

# Function to extract SIFT features from an image


def extract_sift_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Function to perform feature matching between query and database images


def perform_feature_matching(query_descriptors, database_descriptors):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(query_descriptors, database_descriptors, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

# Function to perform image retrieval based on a query image


def perform_image_retrieval(query_image_path, database_path, k=5):
    # Extract SIFT features from the query image
    query_keypoints, query_descriptors = extract_sift_features(
        query_image_path)

    # Load images from the database and compute matches with query image
    image_files = os.listdir(database_path)
    similar_images = []

    for image_file in image_files:
        image_path = os.path.join(database_path, image_file)
        database_keypoints, database_descriptors = extract_sift_features(
            image_path)

        if database_descriptors is not None:
            matches = perform_feature_matching(
                query_descriptors, database_descriptors)
            similar_images.append((image_file, matches))

    # Sort images by number of matches and retrieve top k similar images
    similar_images.sort(key=lambda x: len(x[1]), reverse=True)

    # Display top k similar images
    print("Top", k, "similar images:")
    for image_name, matches in similar_images:
        print(image_name, " - Number of Matches:", len(matches))


# Example usage
query_image_path = "ExploringSimilarityMetrics/test.jpg"
database_path = "ExploringSimilarityMetrics/temp"
perform_image_retrieval(query_image_path, database_path)
