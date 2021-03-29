import face_model
import argparse
import cv2
import os
import time

import numpy as np

from PIL import Image
from detect_keypoints import Handler

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

    print(compare_dict)
    name = max(compare_dict, key=compare_dict.get)
    return name


if __name__ == "__main__":
    vec = args.model.split(',')
    model_prefix = vec[0]
    model_epoch = int(vec[1])
    model = face_model.FaceModel(args.gpu, model_prefix, model_epoch)
    embds_dict = read_embendings("data/", model)

    # img = cv2.imread('1.jpeg')
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    print(frame.shape)

    # for c in range(count):
    while cap.isOpened():
        time_1 = time.time()
        ret, frame = cap.read()
        # ret, frame = True, cv2.imread("/home/dl/0_work/1_LunarEye/4_models/face/mask-detection-and-classification/2.jpg")

        display_img = frame.copy()
        imgs, bboxs = model.get_input(frame)
        handler = Handler('/home/dl/1_study/0_BSU/master_thesis/weights/keypoints/2d106det', 0, ctx_id=0, det_size=640)

        if imgs is not None and bboxs is not None:
            for img, bbox in zip(imgs, bboxs):
                bbox = [int(box) for box in bbox]
                f2 = model.get_feature(img)
                name = compare_emdbs(embds_dict, f2)

                x = bbox[0]
                y = bbox[1]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                color = (0, 0, 255)
                cv2.putText(display_img, name, (x, y - 40), font, font_scale, (0, 255, 0), thickness)
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                img_face = crop_face(frame, bbox)
                temp_image = Image.fromarray(img_face, mode="RGB")
            # print(sim, ": ", sim >= 0.5 and sim < 1.01)

        print("fps: ", 1 / (time.time() - time_1))
        cv2.imshow("0", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()