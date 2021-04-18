import argparse
import cv2
import os

import numpy as np

FONT_SCALE = 0.5
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FACE_MODEL_PATH = "/home/dl/1_study/0_BSU/master_thesis/weights/model-r100-ii/,0"
KEYPOINTS_MODEL_PATH = "/home/dl/1_study/0_BSU/master_thesis/repos/insightface/alignment/coordinateReg/model/2d106det"
EMOTIONS_MODEL_PATH = "/home/dl/1_study/0_BSU/master_thesis/weights/emotion" \
                      "/KerasEffNetB0Gray_128x128_batch_16_withoutDP_ALL_aug_crop_FER_upscale/"
VIDEO_PATH = "/home/dl/1_study/0_BSU/master_thesis/videos/res/6.mp4"

LABELS_DICT_EMO = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}


def parse_args():
    parser = argparse.ArgumentParser(description='face models test')
    parser.add_argument('-is', '--image_size', type=str, required=False, default='112,112',
                        help='PSA user id')
    parser.add_argument('-m', '--model', type=str, required=False, default="",
                        help='path to load model')
    parser.add_argument('-g', '--gpu', type=int, required=False, default=0,
                        help='gpu id')
    parser.add_argument('-w', '--web_cam', help='from web camera', action='store_true')
    parser.add_argument('-md', '--mode', type=str, required=True, default="reco",
                        help='reco (recognition) or emo (emotion)')

    return parser.parse_args()


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
            name, ext = os.path.splitext(img_name)
            if ext != ".npy":
                if not os.path.isfile(os.path.join(path, person_name, name + '.npy')):
                    img = cv2.imread(os.path.join(path, person_name, img_name))
                    imgs, bboxs = model.get_input(img)
                    if imgs is not None:
                        img = imgs[0]

                        f = model.get_feature(img)
                        embds_dict[person_name].append(f)
                        np.save(os.path.join(path, person_name, name + '.npy'), f)
                        print("\033[96m Found: ", img_name, "\033[0m")
                    else:
                        print("\033[93m Not found: ", img_name, "\033[0m")
                else:
                    with open(os.path.join(path, person_name, name + '.npy'), 'rb') as file:
                        f = np.load(file)
                    embds_dict[person_name].append(f)
                    print("\033[92m Load: ", name + ".npy", "\033[0m")

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

    x1, y1 = checked_points[0]
    x2, y2 = checked_points[1]
    x3, y3 = checked_points[2]
    x4, y4 = checked_points[3]

    return x1, y1, x2, y2, x3, y3, x4, y4


def draw_keypoints(display_img, points):
    color = (200, 160, 75)
    for point in points:
        point = np.round(point).astype(np.int)
        for i in range(point.shape[0]):
            p = tuple(point[i])
            cv2.circle(display_img, p, 1, color, 1, cv2.LINE_AA)

    return display_img


def put_text(display_img, text, coords):
    return cv2.putText(display_img, LABELS_DICT_EMO[np.argmax(text[0])], coords, FONT, FONT_SCALE, (0, 255, 0),
                       THICKNESS)