import keras.utils as image
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import VGG19
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os, shutil, glob


# Кластеризація алгоритмом k-means з навчанням без вчителя
def clustering_2(imdir, targetdir, number_clusters):

    image.LOAD_TRUNCATED_IMAGES = True

    model = VGG19(weights='imagenet', include_top=False)

    # Loop over files and get features
    filelist = glob.glob(os.path.join(imdir, '*.jpg'))
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print("    Status: %s / %s" % (i, len(filelist)), end="\r")
        img = image.load_img(imagepath, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = np.array(model.predict(img_data))
        featurelist.append(features.flatten())

    # Clustering
    kmeans = KMeans(n_clusters=number_clusters, init='k-means++', n_init=10, random_state=0).fit(np.array(featurelist))

    # Copy images renamed by cluster
    # Check if target dir exists
    try:
        os.makedirs(targetdir)
    except OSError:
        pass

    # Create a dictionary to store clusters and corresponding image paths
    clusters_dict = {i: [] for i in range(number_clusters)}

    print("\n")
    for i, m in enumerate(kmeans.labels_):
        print("    Copy: %s / %s" % (i, len(kmeans.labels_)), end="\r")
        # Copy with cluster name
        shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")
        # Append image path to the cluster in the dictionary
        clusters_dict[m].append(targetdir + str(m) + "_" + str(i) + ".jpg")

    return clusters_dict


# Function to view images in a cluster
def view_cluster(cluster, clusters_dict):
    plt.figure(figsize=(10, 10))
    # Gets the list of filenames for a cluster
    files = clusters_dict[cluster]
    # Only allow up to {img_num} images to be shown at a time
    img_num = 50
    if len(files) > img_num:
        print(f"Clipping cluster size from {len(files)} to {img_num}")
        files = files[:img_num-1]

    # Plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1)
        img = image.load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

    plt.show()


# ---------------------------------- головні виклики  --------------------------------
if __name__ == "__main__":
    imdir = "C:/Introduction to DataScience/Kachurynets_Lab5/Multi-class Weather Dataset/"
    targetdir = "C:/Introduction to DataScience/Kachurynets_Lab5/Stop_Weather/"
    number_clusters = 3

    # Call the clustering function
    clusters_dict = clustering_2(imdir, targetdir, number_clusters)

    # Call the view_cluster function for each cluster
    for cluster in range(number_clusters):
        view_cluster(cluster, clusters_dict)
