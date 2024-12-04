from scaler import scaler

sc = scaler.Scaler((640, 480))
sc.load_labels("testimg/labels.json")
sc.process_folder("testimg")