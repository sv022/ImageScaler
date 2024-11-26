import cv2
import os
import json
from numpy import around


class Scaler:
    def __init__(self, resoluiton: tuple[int, int], outDir = 'out/'):
        self.resolution = resoluiton
        self.outDir = outDir
        self.labels = {}
        self.labelMap = []

        if not os.path.exists(outDir):
            os.makedirs(outDir)

        with open("config.json") as json_file:
            config = json.load(json_file)
        self.labelMap = config["classes"]
        

    def load_labels(self, labels_path: str):
        with open(labels_path) as json_file:
            labels = json.load(json_file)
        self.labels = labels
    
    def process_image(self, image_path : str, label: bool):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}.")

        resized_image = cv2.resize(image, self.resolution)
        normalized_image = around(resized_image / 255.0, decimals=3)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        labelName = self.labels[image_name]

        with open(f"{self.outDir}{image_name}.txt", "w") as output_file:
            flattened_values = normalized_image.flatten()
            output_file.write(" ".join(map(str, flattened_values)) + '\n')
            if label:
                output_file.write(" ".join(["1" if l == labelName else "0" for l in self.labelMap]))

        # print(f"Значения для {image_name} записаны в {self.outDir}")


    def process_folder(self, input_folder : str, label = True):
        if not os.path.isdir(input_folder):
            raise PermissionError(f"Папка {input_folder} не существует.")
        
        if not self.labels and label:
            raise ValueError("Can not label images - labels not set. Try using load_labels() first.")

        for file_name in os.listdir(input_folder):
            if file_name.lower().endswith(".jpg"):
                image_path = os.path.join(input_folder, file_name)
                self.process_image(image_path, label)