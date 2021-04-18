import face_model
import cv2
import os
import time
import keras

import tensorflow as tf
import numpy as np

from detect_keypoints import Handler
from utils import LABELS_DICT_EMO, FACE_MODEL_PATH, KEYPOINTS_MODEL_PATH, EMOTIONS_MODEL_PATH, VIDEO_PATH, test_points, \
    parse_args, read_embendings, compare_emdbs, draw_keypoints, put_text

from tensorflow.compat.v1.keras.backend import set_session

args = parse_args()

# setting initialization
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

if __name__ == "__main__":
    vec = FACE_MODEL_PATH.split(",")
    model_prefix = vec[0]
    model_epoch = int(vec[1])
    model = face_model.FaceModel(args.gpu, model_prefix, model_epoch)

    if args.mode == "emo":
        handler = Handler(KEYPOINTS_MODEL_PATH, 0, ctx_id=0, det_size=640)
        clf = keras.models.load_model(EMOTIONS_MODEL_PATH)
    elif args.mode == "reco":
        embds_dict = read_embendings("data/", model)

    if args.web_cam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        time_1 = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(frame, (740, 640))

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

                if args.mode == "emo":
                    points = handler.get(frame, bbox, get_all=True)

                    display_img = draw_keypoints(display_img, points)

                    points = points[0]

                    # crop stage
                    x1, y1 = int(np.min(points[:, 0])), int(np.min(points[:, 1]))
                    x2, y2 = int(np.max(points[:, 0])), int(np.min(points[:, 1]))
                    x3, y3 = int(np.min(points[:, 0])), int(np.max(points[:, 1]))
                    x4, y4 = int(np.max(points[:, 0])), int(np.max(points[:, 1]))
                    x1, y1, x2, y2, x3, y3, x4, y4 = test_points([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], frame.shape)

                    crop_img = frame[min(y1, y2):max(y3, y4), min(x1, x3):max(x2, x4)]

                    # if problems with keypoints detection
                    if crop_img.size == 0:
                        points = test_points([[x, y], [x + w, y + h]], frame.shape)
                        crop_img = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]\

                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    crop_img = cv2.resize(crop_img, (128, 128))
                    crop_img /= 255.

                    emo = clf.predict(np.expand_dims(crop_img, 0))
                    display_img = put_text(display_img, LABELS_DICT_EMO[np.argmax(emo[0])], (x, y - 20))

                elif args.mode == "reco":
                    f2 = model.get_feature(img)
                    name = compare_emdbs(embds_dict, f2)
                    display_img = put_text(display_img, name, (x, y - 40))

            # print(sim, ": ", sim >= 0.5 and sim < 1.01)

        print("fps: ", 1 / (time.time() - time_1))
        cv2.imshow("frame", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
