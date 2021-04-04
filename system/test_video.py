import face_model
import argparse
import cv2
import os
import time
import keras

import tensorflow as tf
import numpy as np

from PIL import Image
from detect_keypoints import Handler
from tensorflow.compat.v1.keras.backend import set_session

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
args = parser.parse_args()

font_scale = 1
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

labels_dict_fer = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}


def crop_face(img, _box):
    h = _box[3] - _box[1]
    w = _box[2] - _box[0]

    proba_h = 0.1  # random.choice([0.4, 0.45, 0.5, 0.55, 0.6])
    proba_w = 0.1  # random.choice([0.2, 0.25, 0.3, 0.35, 0.4])
    indent_h = int(h * proba_h)
    indent_w = int(w * proba_w)

    if _box[0] - indent_w >= 0:
        img_face = img[:, (_box[0] - indent_w):(_box[2] + indent_w), :]
    else:
        img_face = img[:, :(_box[2] + indent_w), :]

    if _box[1] - indent_h >= 0:
        img_face = img_face[(_box[1] - indent_h):(_box[3] + indent_h), :, :]
    else:
        img_face = img_face[:(_box[3] + indent_h), :, :]

    return img_face


def read_embendings(path, model):

    person_names = os.listdir(path)
    embds_dict = {person_name: [] for person_name in person_names}
    for person_name in person_names:
        print("Processing: ", person_name)
        for img_name in os.listdir(os.path.join(path, person_name)):
            img = cv2.imread(os.path.join(path, person_name, img_name))
            imgs, bboxs = model.get_input(img)
            if imgs is not None and bboxs is not None:
                img = imgs[0]
                bbox = bboxs[0]
                print("\033[96m Found: ", img_name, "\033[0m")
                f = model.get_feature(img)
                embds_dict[person_name].append(f)
            else:
                print("\033[93m Not found: ", img_name, "\033[0m")
                continue

    return embds_dict


def compare_emdbs(data_dict, embd):

    compare_dict = {}
    for person_name, embds in data_dict.items():
        compare_dict[person_name] = np.mean(np.dot(embds, embd))

    name = max(compare_dict, key=compare_dict.get)
    return name


def test_points(points, shape):
    checked_points = []
    for point in points:
        checked_point = point

        if point[0] < 0:
            checked_point[0] = 1
        if point[0] > shape[1]:
            checked_point[0] = shape[1] - 1

        if point[1] < 0:
            checked_point[1] = 1
        if point[1] > shape[0]:
            checked_point[1] = shape[0] - 1

        checked_points.append(checked_point)

    return checked_points


if __name__ == "__main__":
    vec = args.model.split(',')
    model_prefix = vec[0]
    model_epoch = int(vec[1])
    model = face_model.FaceModel(args.gpu, model_prefix, model_epoch)
    # embds_dict = read_embendings("data/", model)

    handler = Handler('/home/dl/1_study/0_BSU/master_thesis/repos/insightface/alignment/coordinateReg/model/2d106det',
                      0, ctx_id=0, det_size=640)

    clf = keras.models.load_model("/home/dl/1_study/0_BSU/master_thesis/weights/emotion/KerasEffNetB0Gray_128x128_batch_16_withoutDP_ALL_aug_crop_FREP_upscale/")
    temp = np.zeros((128, 128, 1))
    clf.predict(np.expand_dims(temp, 0))

    # img = cv2.imread('1.jpeg')
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/home/dl/1_study/0_BSU/master_thesis/videos/merge/4_Bubbles-62455.mp4")
    ret, frame = cap.read()
    print(frame.shape)

    # for c in range(count):
    while cap.isOpened():
        time_1 = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 640))
        # ret, frame = True, cv2.imread("/home/dl/0_work/1_LunarEye/4_models/face/mask-detection-and-classification/2.jpg")

        display_img = frame.copy()
        imgs, bboxs = model.get_input(frame)

        if imgs is not None and bboxs is not None:
            for img, bbox in zip(imgs, bboxs):
                bbox = np.array([int(box) for box in bbox])

                x = bbox[0]
                y = bbox[1]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                points = handler.get(frame, bbox, get_all=True)

                # color = (200, 160, 75)
                # for point in points:
                #     point = np.round(point).astype(np.int)
                #     for i in range(point.shape[0]):
                #         p = tuple(point[i])
                #         cv2.circle(display_img, p, 1, color, 1, cv2.LINE_AA)

                points = points[0]

                # crop stage
                x1, y1 = int(np.min(points[:, 0])), int(np.min(points[:, 1]))
                x2, y2 = int(np.max(points[:, 0])), int(np.min(points[:, 1]))
                x3, y3 = int(np.min(points[:, 0])), int(np.max(points[:, 1]))
                x4, y4 = int(np.max(points[:, 0])), int(np.max(points[:, 1]))

                points = test_points([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], frame.shape)
                #
                x1, y1 = points[0]
                x2, y2 = points[1]
                x3, y3 = points[2]
                x4, y4 = points[3]
                #
                crop_img = frame[min(y1, y2):max(y3, y4), min(x1, x3):max(x2, x4)]
                if crop_img.size == 0:
                    print(x, y, x+w, y+h)
                    points = test_points([[x, y], [x+w, y+h]], frame.shape)
                    print(points)
                    crop_img = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                crop_img = cv2.resize(crop_img, (128, 128))
                crop_img /= 255.
                emo = clf.predict(np.expand_dims(crop_img, 0))
                cv2.putText(display_img, labels_dict_fer[np.argmax(emo[0])], (x, y - 20),
                            font, font_scale, (0, 255, 0), thickness)

                # f2 = model.get_feature(img)
                # name = compare_emdbs(embds_dict, f2)
                # cv2.putText(display_img, name, (x, y - 40), font, font_scale, (0, 255, 0), thickness)

                # img_face = crop_face(frame, bbox)
                # temp_image = Image.fromarray(img_face, mode="RGB")
            # print(sim, ": ", sim >= 0.5 and sim < 1.01)

        print("fps: ", 1 / (time.time() - time_1))
        cv2.imshow("0", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()