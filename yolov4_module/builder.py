from core.yolov4 import YOLO, decode_train, compute_loss
import tensorflow as tf
from core import utils
from core.utils import freeze_all, unfreeze_all
import numpy as np
import os
import cv2
import random

BATCH_SIZE = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
INPUT_SIZE = 416
LR_INIT = 1e-3
LR_END = 1e-6
WARMUP_EPOCHS = 2
FISRT_STAGE_EPOCHS = 20
SECOND_STAGE_EPOCHS = 30

ANCHORS = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
ANCHORS_V3 = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
ANCHORS_TINY = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
STRIDES = [8, 16, 32]
STRIDES_TINY = [16, 32]
XYSCALE = [1.2, 1.1, 1.05]
XYSCALE_TINY = [1.05, 1.05]
ANCHOR_PER_SCALE = 3
IOU_LOSS_THRESH = 0.5


class Yolo:
    def __init__(self, base_path, class_path, annotation_train, annotation_test, weight_path):
        self.isfreeze = False

        self.CLASSES = class_path
        self.trainset = Dataset(self.CLASSES, annotation_train, is_training=True)
        self.testset = Dataset(self.CLASSES, annotation_test, is_training=False)
        self.steps_per_epoch = len(self.trainset)
        self.first_stage_epochs = FISRT_STAGE_EPOCHS
        self.second_stage_epochs = SECOND_STAGE_EPOCHS
        self.optimizer = tf.keras.optimizers.Adam()

        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = WARMUP_EPOCHS * self.steps_per_epoch
        self.total_steps = (self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_epoch

        self.freeze_layers = utils.load_freeze_layer("yolov4", False)

        self.STRIDES, self.ANCHORS, self.NUM_CLASS, self.XYSCALE = self.load_config(self.CLASSES)
        self.IOU_LOSS_THRESH = IOU_LOSS_THRESH
        self.base_path = base_path

        self.writer = tf.summary.create_file_writer("{}/{}".format(self.base_path, "log"))

        self.model = self.build_model(weight_path)

    @staticmethod
    def load_config(CLASSES):
        RETURN_STRIDES = np.array(STRIDES)
        RETURN_ANCHORS = utils.get_anchors(ANCHORS, False)
        RETURN_XYSCALE = XYSCALE
        RETURN_NUM_CLASS = len(utils.read_class_names(CLASSES))

        return RETURN_STRIDES, RETURN_ANCHORS, RETURN_NUM_CLASS, RETURN_XYSCALE

    def build_model(self, weight=None):
        input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])

        feature_maps = YOLO(input_layer, self.NUM_CLASS, "yolov4", False)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, INPUT_SIZE // 8, self.NUM_CLASS, self.STRIDES, self.ANCHORS, i, self.XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, INPUT_SIZE // 16, self.NUM_CLASS, self.STRIDES, self.ANCHORS, i, self.XYSCALE)
            else:
                bbox_tensor = decode_train(fm, INPUT_SIZE // 32, self.NUM_CLASS, self.STRIDES, self.ANCHORS, i, self.XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

        model = tf.keras.Model(input_layer, bbox_tensors)

        if weight is not None:
            model.load_weights(weight)

        return model

    def train_step(self, image_data, target):
        with tf.GradientTape() as tape:
            pred_result = self.model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(self.freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=self.STRIDES,
                                          NUM_CLASS=self.NUM_CLASS, IOU_LOSS_THRESH=self.IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (self.global_steps, self.total_steps, self.optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            self.global_steps.assign_add(1)
            if self.global_steps < self.warmup_steps:
                lr = self.global_steps / self.warmup_steps * LR_INIT
            else:
                lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
                    (1 + tf.cos((self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi))
                )
            self.optimizer.lr.assign(lr.numpy())

            # writing summary data
            with self.writer.as_default():
                tf.summary.scalar("lr", self.optimizer.lr, step=self.global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=self.global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=self.global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=self.global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=self.global_steps)
            self.writer.flush()

    def test_step(self, image_data, target):
        with tf.GradientTape() as tape:
            pred_result = self.model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(self.freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=self.STRIDES,
                                          NUM_CLASS=self.NUM_CLASS, IOU_LOSS_THRESH=self.IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (self.global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))

    def train(self):

        for epoch in range(self.first_stage_epochs + self.second_stage_epochs):
            if epoch < self.first_stage_epochs:
                if not self.isfreeze:
                    self.isfreeze = True
                    for name in self.freeze_layers:
                        freeze = self.model.get_layer(name)
                        freeze_all(freeze)
            elif epoch >= self.first_stage_epochs:
                if self.isfreeze:
                    self.isfreeze = False
                    for name in self.freeze_layers:
                        freeze = self.model.get_layer(name)
                        unfreeze_all(freeze)
            for image_data, target in self.trainset:
                self.train_step(image_data, target)
            for image_data, target in self.testset:
                self.test_step(image_data, target)
            self.model.save_weights("{}/checkpoints/yolov4".format(self.base_path))


class Dataset(object):
    """implement Dataset here"""

    def __init__(self, CLASSES, annotation_path: str, is_training: bool, dataset_type: str = "converted_coco"):
        self.tiny = False
        self.strides, self.anchors, NUM_CLASS, XYSCALE = Yolo.load_config(CLASSES)
        self.dataset_type = dataset_type

        self.annot_path = annotation_path
        self.input_sizes = (
            INPUT_SIZE
        )
        self.batch_size = (
            BATCH_SIZE
        )
        self.data_aug = True if is_training else False

        self.train_input_sizes = INPUT_SIZE
        self.classes = utils.read_class_names(CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        with open(self.annot_path, "r") as f:
            txt = f.readlines()
            if self.dataset_type == "converted_coco":
                annotations = [
                    line.strip()
                    for line in txt
                    if len(line.strip().split()[1:]) != 0
                ]
            elif self.dataset_type == "yolo":
                annotations = []
                for line in txt:
                    image_path = line.strip()
                    root, _ = os.path.splitext(image_path)
                    with open(root + ".txt") as fd:
                        boxes = fd.readlines()
                        string = ""
                        for box in boxes:
                            box = box.strip()
                            box = box.split()
                            class_num = int(box[0])
                            center_x = float(box[1])
                            center_y = float(box[2])
                            half_width = float(box[3]) / 2
                            half_height = float(box[4]) / 2
                            string += " {},{},{},{},{}".format(
                                center_x - half_width,
                                center_y - half_height,
                                center_x + half_width,
                                center_y + half_height,
                                class_num,
                            )
                        annotations.append(image_path + string)

        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/cpu:0"):
            # self.train_input_size = random.choice(self.train_input_sizes)
            self.train_input_size = INPUT_SIZE
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    3,
                ),
                dtype=np.float32,
            )

            batch_label_sbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[0],
                    self.train_output_sizes[0],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_mbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[1],
                    self.train_output_sizes[1],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_lbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[2],
                    self.train_output_sizes[2],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )

            batch_sbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )
            batch_mbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )
            batch_lbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    (
                        label_sbbox,
                        label_mbbox,
                        label_lbbox,
                        sbboxes,
                        mbboxes,
                        lbboxes,
                    ) = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                return (
                    batch_image,
                    (
                        batch_smaller_target,
                        batch_medium_target,
                        batch_larger_target,
                    ),
                )
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            )
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            )
            crop_xmax = max(
                w, int(max_bbox[2] + random.uniform(0, max_r_trans))
            )
            crop_ymax = max(
                h, int(max_bbox[3] + random.uniform(0, max_d_trans))
            )

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        if self.dataset_type == "converted_coco":
            bboxes = np.array(
                [list(map(int, box.split(","))) for box in line[1:]]
            )
        elif self.dataset_type == "yolo":
            height, width, _ = image.shape
            bboxes = np.array(
                [list(map(float, box.split(","))) for box in line[1:]]
            )
            bboxes = bboxes * np.array([width, height, width, height, 1])
            bboxes = bboxes.astype(np.int64)

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(
                np.copy(image), np.copy(bboxes)
            )
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(
                np.copy(image), np.copy(bboxes)
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preprocess(
            np.copy(image),
            [self.train_input_size, self.train_input_size],
            np.copy(bboxes),
        )
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(3)
        ]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )
            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = utils.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(
                    bbox_count[best_detect] % self.max_bbox_per_scale
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
