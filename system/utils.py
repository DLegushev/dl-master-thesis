import argparse
import cv2
import os

import numpy as np

FONT_SCALE = 0.5
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FACE_MODEL_PATH = "/home/dl/1_study/0_BSU/master_thesis/weights/model-r100-ii/,0"
KEYPOINTS_MODEL_PATH = "/home/dl/1_study/0_BSU/master_thesis/repos/insightface/alignment/coordinateReg/model/2d106det"
EMOTIONS_MODEL_PATH_CROP = "/home/dl/1_study/0_BSU/master_thesis/weights/emotion" \
                           "/KerasEffNetB0Gray_128x128_batch_16_withoutDP_ALL_aug_crop_FER_upscale/"
EMOTIONS_MODEL_PATH_DEWARP = "/home/dl/1_study/0_BSU/master_thesis/weights/emotion" \
                             "/KerasEffNetB0Gray_128x128_batch_16_withoutDP_ALL_aug_dewarp_FER_upscale/"
VIDEO_PATH = "/home/dl/1_study/0_BSU/master_thesis/videos/res/9.mp4"

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
    parser.add_argument('-pp', '--preprocess', type=str, required=True, default="crop",
                        help='crop or dewarp')

    return parser.parse_args()


def get_model_name(args):
    if args.preprocess == "crop":
        return EMOTIONS_MODEL_PATH_CROP
    elif args.preprocess == "dewarp":
        return EMOTIONS_MODEL_PATH_DEWARP
    else:
        raise RuntimeError("preprocess name error")


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

        checked_points.append(checked_point[0])
        checked_points.append(checked_point[1])

    return checked_points


def draw_keypoints(display_img, points):
    color = (200, 160, 75)
    for point in points:
        point = np.round(point).astype(np.int)
        for i in range(point.shape[0]):
            p = tuple(point[i])
            cv2.circle(display_img, p, 1, color, 1, cv2.LINE_AA)

    return display_img


def put_text(display_img, text, coords):
    return cv2.putText(display_img, text, coords, FONT, FONT_SCALE, (0, 255, 0),
                       THICKNESS)


def four_point_transform(image, pts):
    rect = np.array((pts[0], pts[1], pts[3], pts[2]))
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # constract the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [0, maxHeight],
        [maxWidth, maxHeight]], dtype="float32")
    # dst = np.array([
    #     [int(maxWidth/3), int(maxHeight/3)],
    #     [int(maxWidth*2/3), int(maxHeight/3)],
    #     [int(maxWidth/2), int(maxHeight/2)],
    #     [int(maxWidth/2), maxHeight]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    # M = cv2.getPerspectiveTransform(rect, dst)
    h, mask = cv2.findHomography(pts, dst, cv2.RANSAC)
    height, width, channels = image.shape
    warped = cv2.warpPerspective(image, h, (maxWidth, maxHeight))

    # return the warped image
    return warped


def preprocess(args, points, frame, bbox):
    if args.preprocess == "crop":
        # crop stage
        x1, y1 = int(np.min(points[:, 0])), int(np.min(points[:, 1]))
        x2, y2 = int(np.max(points[:, 0])), int(np.min(points[:, 1]))
        x3, y3 = int(np.min(points[:, 0])), int(np.max(points[:, 1]))
        x4, y4 = int(np.max(points[:, 0])), int(np.max(points[:, 1]))
        x1, y1, x2, y2, x3, y3, x4, y4 = test_points([[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                                                     frame.shape)

        crop_img = frame[min(y1, y2):max(y3, y4), min(x1, x3):max(x2, x4)]

        # if problems with keypoints detection
        if crop_img.size == 0:
            points = test_points([[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]], frame.shape)
            crop_img = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]

        cv2.imshow("crop", crop_img)
        preproc_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        preproc_img = cv2.resize(preproc_img, (128, 128))
        preproc_img /= 255.
    elif args.preprocess == "dewarp":
        x1_crop, y1_crop, x2_crop, y2_crop = test_points([[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]],
                                                         frame.shape)
        crop_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]

        # dewarp stage
        x1_dewarp, y1_dewarp = int(points[48][0] - x1_crop), int(points[48][1] - y1_crop)
        x2_dewarp, y2_dewarp = int(points[105][0] - x1_crop), int(points[105][1] - y1_crop)
        x3_dewarp, y3_dewarp = int(points[5][0] - x1_crop), int(points[5][1] - y1_crop)
        x4_dewarp, y4_dewarp = int(points[21][0] - x1_crop), int(points[21][1] - y1_crop)

        x1_dewarp, y1_dewarp, x2_dewarp, y2_dewarp, x3_dewarp, y3_dewarp, x4_dewarp, y4dewarp = \
            test_points([[x1_dewarp, y1_dewarp],
                         [x2_dewarp, y2_dewarp],
                         [x3_dewarp, y3_dewarp],
                         [x4_dewarp, y4_dewarp]], frame.shape)

        dewarp_img = four_point_transform(crop_img, np.array([[x1_dewarp, y1_dewarp],
                                                              [x2_dewarp, y2_dewarp],
                                                              [x3_dewarp, y3_dewarp],
                                                              [x4_dewarp, y4_dewarp]]))

        # if problems with keypoints detection
        if dewarp_img.size == 0:
            dewarp_img = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]

        cv2.imshow("dewarp", dewarp_img)
        preproc_img = cv2.cvtColor(dewarp_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        preproc_img = cv2.resize(preproc_img, (128, 128))
        preproc_img /= 255.
    else:
        raise RuntimeError("preprocess name error")

    return preproc_img
