from scaler import scaler
import json


def main():
    with open('config.json') as json_file:
        config = json.load(json_file)

    sc = scaler.Scaler(config["dimesions"])
    sc.load_labels(config["labelsPath"], config["classes"])
    sc.process_folder(config["datasetPath"])


if __name__ == "__main__":
    main()