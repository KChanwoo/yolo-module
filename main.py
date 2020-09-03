from yolov4_module.builder import Yolo, INPUT_SIZE
import cv2
import numpy as np
import tensorflow as tf
from core import utils

yolo = Yolo("./", "./data/classes/graph.names", "./data/dataset/graph.txt"
            , "./data/dataset/graph_test.txt", "./checkpoints/yolov4")

# yolo.train()
image = cv2.imread("/Users/gwonchan-u/Downloads/Infant ABR 51-110/I52.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
image_data = image_data / 255.
# image_data = image_data[np.newaxis, ...].astype(np.float32)

images_data = []
for i in range(1):
    images_data.append(image_data)
images_data = np.asarray(images_data).astype(np.float32)

batch_data = tf.constant(images_data)
pred_bbox = yolo.model(batch_data, training=False)
for key, value in pred_bbox.items():
    boxes = value[:, :, 0:4]
    pred_conf = value[:, :, 4:]

boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )
pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
image = utils.draw_bbox(image, pred_bbox)
# image = utils.draw_bbox(image_data*255, pred_bbox)
image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
cv2.imwrite("./test.png", image)
