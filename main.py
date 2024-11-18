import os
import cv2
import numpy as np
from secrets import choice
from skimage.morphology import skeletonize
from skimage.feature import canny
from sklearn.metrics import pairwise_distances

# Utility functions to load and preprocess fingerprint images
def load_and_preprocess_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image

# Method 1: Minutiae Detection
def extract_minutiae_features(image):
    skeleton = skeletonize(image > 127)
    minutiae = []
    for i in range(1, skeleton.shape[0] - 1):
        for j in range(1, skeleton.shape[1] - 1):
            if skeleton[i, j]:
                neighbors = skeleton[i-1:i+2, j-1:j+2].sum() - skeleton[i, j]
                if neighbors == 1:
                    minutiae.append((i, j))
                elif neighbors > 2:
                    minutiae.append((i, j))
    return minutiae

# Method 2: Fourier Transform Features
def extract_fourier_features(image):
    f_transform = np.abs(np.fft.fft2(image))
    return f_transform.ravel()[:256]  # Truncate to 256 elements for comparison

# Method 3: ORB Feature Detection and Description
def extract_orb_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

# Feature Matching functions
def match_minutiae(features1, features2):
    dist_matrix = pairwise_distances(features1, features2)
    threshold = 10
    matches = np.sum(dist_matrix < threshold)
    return matches / min(len(features1), len(features2))

def match_fourier_features(features1, features2):
    return np.linalg.norm(features1 - features2)

def match_orb_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return len(matches)

# Run methods on a pair of images
def evaluate_methods(image1, image2):
    # Minutiae-based matching
    minutiae1 = extract_minutiae_features(image1)
    minutiae2 = extract_minutiae_features(image2)
    minutiae_score = match_minutiae(minutiae1, minutiae2)

    # Fourier transform-based matching
    fourier_features1 = extract_fourier_features(image1)
    fourier_features2 = extract_fourier_features(image2)
    fourier_score = match_fourier_features(fourier_features1, fourier_features2)

    # ORB descriptor matching
    orb_features1 = extract_orb_features(image1)
    orb_features2 = extract_orb_features(image2)
    orb_score = match_orb_features(orb_features1, orb_features2) if orb_features1 is not None and orb_features2 is not None else 0

    return {
        "minutiae_score": minutiae_score,
        "fourier_score": fourier_score,
        "orb_score": orb_score
    }

# Evaluate Hybrid Decision (majority vote)
def hybrid_system(minutiae_score, fourier_score, orb_score, minutiae_thresh=0.5, fourier_thresh=5000, orb_thresh=30):
    decisions = [
        bool(minutiae_score >= minutiae_thresh),
        bool(fourier_score <= fourier_thresh),
        bool(orb_score >= orb_thresh)
    ]
    return decisions

# Load and evaluate all pairs in a directory
def evaluate_all_pairs(directory, thresholds, run_num):
    avg_1_true = 0
    avg_2_true = 0
    avg_3_true = 0
    avg_1_false = 0
    avg_2_false = 0
    avg_3_false = 0
    matches_1_true = 0
    matches_2_true = 0
    matches_3_true = 0
    matches_1_false = 0
    matches_2_false = 0
    matches_3_false = 0
    matches_1_c = 0
    matches_2_c = 0
    matches_3_c = 0
    total_pairs = 0

    for i in range(1, 1501):  # 1500 training pairs
        if i % 100 == 0:
            print('Evaluating pair ', i, '/1500')
        image1_path = os.path.join(directory, f"f{i:04}.png")
        image2_path = os.path.join(directory, f"s{i:04}.png")
        rand_img_num = i
        while rand_img_num == i:
            rand_img_num = choice(range(1, 1501))
        image3_path = os.path.join(directory, f"f{rand_img_num:04}.png")

        # Load images
        image1 = load_and_preprocess_image(image1_path)
        image2 = load_and_preprocess_image(image2_path)
        image3 = load_and_preprocess_image(image3_path)

        # Evaluate individual methods
        results = evaluate_methods(image1, image2)
        avg_1_true += results['minutiae_score']
        avg_2_true += results['fourier_score']
        avg_3_true += results['orb_score']

        decision = hybrid_system(
            results['minutiae_score'], results['fourier_score'], results['orb_score'],
            thresholds['minutiae'], thresholds['fourier'], thresholds['orb']
        )

        results2 = evaluate_methods(image1, image3)
        avg_1_false += results['minutiae_score']
        avg_2_false += results['fourier_score']
        avg_3_false += results['orb_score']

        decision2 = hybrid_system(
            results2['minutiae_score'], results2['fourier_score'], results2['orb_score'],
            thresholds['minutiae'], thresholds['fourier'], thresholds['orb']
        )

        # Count matches
        if decision[0]:
            matches_1_true += 1
            matches_1_c += 1
        if decision[1]:
            matches_2_true += 1
            matches_2_c += 1
        if decision[2]:
            matches_3_true += 1
            matches_3_c += 1
        if decision2[0]:
            matches_1_false += 1
        else:
            matches_1_c += 1
        if decision2[1]:
            matches_2_false += 1
        else:
            matches_2_c += 1
        if decision2[2]:
            matches_3_false += 1
        else:
            matches_3_c += 1

        total_pairs += 1

    if run_num == 0:
        return avg_1_true / total_pairs, avg_1_false / total_pairs, avg_2_true / total_pairs, avg_2_false / total_pairs, avg_3_true / total_pairs, avg_3_false / total_pairs
    else:
        print('minutiae:', 'FAR:', matches_1_false / total_pairs, ',', 'FRR:', 1 - matches_1_true / total_pairs, ',', 'EER:', 1 - matches_1_c / 2 / total_pairs)
        print('fourier:', 'FAR:', matches_2_false / total_pairs, ',', 'FRR:', 1 - matches_2_true / total_pairs, ',', 'EER:', 1 - matches_2_c / 2 / total_pairs)
        print('orb:', 'FAR:', matches_3_false / total_pairs, ',', 'FRR:', 1 - matches_3_true / total_pairs, ',', 'EER:', 1 - matches_3_c / 2 / total_pairs)

# Evaluate on test set
def evaluate_test_pairs(test_directory, thresholds):
    matches = 0
    fas = 0
    frs = 0
    total_pairs = 0

    for i in range(1501, 2001):  # 500 test pairs
        if i % 100 == 0:
            print('Evaluating pair ', i, '/2000')
        actual_image = choice([True, False])
        image1_path = os.path.join(test_directory, f"f{i:04}.png")
        image2_path = os.path.join(test_directory, f"s{i:04}.png")
        if not actual_image:
            rand_num = choice(range(1501, 2001))
            image2_path = os.path.join(test_directory, f"s{rand_num:04}.png")

        # Load images
        image1 = load_and_preprocess_image(image1_path)
        image2 = load_and_preprocess_image(image2_path)

        # Evaluate individual methods
        results = evaluate_methods(image1, image2)

        # Use hybrid system for decision
        if actual_image:
            avg_1_true += results['minutiae_score']
            avg_2_true += results['fourier_score']
            avg_3_true += results['orb_score']
        else:
            avg_1_false += results['minutiae_score']
            avg_2_false += results['fourier_score']
            avg_3_false += results['orb_score']
        decisions = hybrid_system(
            results['minutiae_score'], results['fourier_score'], results['orb_score'],
            thresholds['minutiae'], thresholds['fourier'], thresholds['orb']
        )
        decision = sum(decisions) >= 2 # Get majority result

        # Count correct matches
        if (decision and actual_image) or (not decision and not actual_image):
            matches += 1
        if (decision and not actual_image):
            fas += 1
        if (not decision and actual_image):
            frs += 1
        total_pairs += 1

    print('hybrid:', 'fas:', fas / total_pairs, ',', 'frs:', frs / total_pairs, ',', 'eer:', 1 - matches / total_pairs)

# Main Function
if __name__ == "__main__":
    # Define thresholds based on experimentation
    thresholds = {
        'minutiae': 2.92,
        'fourier': 3450000.0,
        'orb': 80.0
    }

    # Set directories for training and testing
    train_directory = "/home/student/Documents/training/"
    test_directory = "/home/student/Documents/test/"

    # Train on all 1500 pairs
    print("Evaluating on Training Data...")
    avg_1_true2, avg_1_false2, avg_2_true2, avg_2_false2, avg_3_true2, avg_3_false2 = evaluate_all_pairs(train_directory, thresholds, 0)

    thresholds = {
        'minutiae': avg_1_false2,
        'fourier': avg_2_false2,
        'orb': avg_3_false2
    }
    evaluate_all_pairs(train_directory, thresholds, 1)
    thresholds = {
        'minutiae': (avg_1_false2 + avg_1_true2) / 2,
        'fourier': (avg_2_false2 + avg_2_true2) / 2,
        'orb': (avg_3_false2 + avg_3_true2) / 2
    }
    evaluate_all_pairs(train_directory, thresholds, 2)
    thresholds = {
        'minutiae': avg_1_true2,
        'fourier': avg_2_true2,
        'orb': avg_3_true2
    }
    evaluate_all_pairs(train_directory, thresholds, 3)

    # Test on 500 additional pairs
    thresholds = {
        'minutiae': (avg_1_false2 + avg_1_true2) / 2,
        'fourier': (avg_2_false2 + avg_2_true2) / 2,
        'orb': (avg_3_false2 + avg_3_true2) / 2
    }
    print("\nEvaluating on Test Data...")
    evaluate_test_pairs(test_directory, thresholds)
    evaluate_test_pairs(test_directory, thresholds)
    evaluate_test_pairs(test_directory, thresholds)

main.py
Displaying main.py.
