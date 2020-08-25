from core.builder import Yolo

yolo = Yolo("./", "./data/classes/graph.names", "./data/dataset/graph.txt", "./data/dataset/graph_test.txt", None)

yolo.train()
