import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import pprint
from color_classifier import ColorClassifier

class SkinDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.color_classifier = ColorClassifier()

    def extract_skin(self):
        img = self.image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_threshold = np.array([2, 40, 2], dtype=np.uint8)
        upper_threshold = np.array([25, 255, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(img, lower_threshold, upper_threshold)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        skin = cv2.bitwise_and(img, img, mask=skin_mask)

        detect_skin = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
        return detect_skin

    def remove_black(self, estimator_labels, estimator_cluster):
        has_black = False
        occurance_counter = Counter(estimator_labels)

        for x in occurance_counter.most_common(len(estimator_cluster)):
            color = [int(i) for i in estimator_cluster[x[0]].tolist()]

            if Counter(color) == Counter([0, 0, 0]):
                del occurance_counter[x[0]]
                has_black = True
                estimator_cluster = np.delete(estimator_cluster, x[0], 0)
                break

        return occurance_counter, estimator_cluster, has_black

    def get_color_information(self, estimator_labels, estimator_cluster, has_thresholding=False):
        occurance_counter = None
        color_information = []
        has_black = False

        if has_thresholding:
            occurance, cluster, black = self.remove_black(estimator_labels, estimator_cluster)
            occurance_counter = occurance
            estimator_cluster = cluster
            has_black = black
        else:
            occurance_counter = Counter(estimator_labels)

        total_occurance = sum(occurance_counter.values())

        for x in occurance_counter.most_common(len(estimator_cluster)):
            index = int(x[0])
            index = (index-1) if (has_thresholding and has_black and (int(index) != 0)) else index
            color = estimator_cluster[index].tolist()
            color_percentage = (x[1] / total_occurance)
            color_info = {"cluster_index": index, "color": color, "color_percentage": color_percentage}
            color_information.append(color_info)

        return color_information

    def extract_dominant_color(self, number_of_colors=5, has_thresholding=False):
        if has_thresholding:
            number_of_colors += 1

        img = self.image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1]), 3)

        estimator = KMeans(n_clusters=number_of_colors, random_state=0)
        estimator.fit(img)

        color_information = self.get_color_information(estimator.labels_, estimator.cluster_centers_, has_thresholding)
        return color_information

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]

    def rgb_to_hex(self, rgb):
        # Ensure the RGB values are in the valid range (0 to 255)
        rgb = [max(0, min(255, x)) for x in rgb]

        # Convert RGB to hexadecimal
        hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

        return hex_color

    def plot_color_bar(self, color_information):
        color_bar = np.zeros((100, 500, 3), dtype="uint8")
        top_x = 0

        for x in color_information:
            bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])
            color = tuple(map(int, (x['color'])))
            cv2.rectangle(color_bar, (int(top_x), 0), (int(bottom_x), color_bar.shape[0]), color, -1)
            top_x = bottom_x

        return color_bar

    def pretty_print_data(self, color_info):
        for x in color_info:
            print(pprint.pformat(x))
            print()

    def process_image(self, number_of_colors=5, has_thresholding=False):
        skin = self.extract_skin()
        dominant_colors = self.extract_dominant_color(number_of_colors, has_thresholding)

        rgb_int = [round(x) for x in dominant_colors[0]['color']]
        if(rgb_int[0] < 30 and rgb_int[1] < 30 and rgb_int[2] < 30):
            rgb_int = [round(x) for x in dominant_colors[1]['color']]

        hex_color = self.rgb_to_hex(rgb_int)
        #print(hex_color)
        #self.pretty_print_data(dominant_colors)
        return self.color_classifier.predict_color_type(hex_color)
