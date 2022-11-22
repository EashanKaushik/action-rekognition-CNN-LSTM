import json
import os
import cv2
import numpy as np
import glob
import mediapipe as mp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
SEED = 42
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64


def mediapipe_detection(image, model):
    """
    # results.face_landmarks.landmark  # 468 #  Error if not present
    # results.pose_landmarks.landmark  # 33
    # results.left_hand_landmarks.landmaark  # 21 #  Error if not present
    # results.right_hand_landmarks.landmaark  # 21 #  Error if not present
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def extract_keypoints(results):

    try:
        face = np.array([[res.x, res.y, res.z]
                        for res in results.face_landmarks.landmark]).flatten()
    except Exception as ex:
        face = np.zeros(468*3)
    try:
        pose = np.array([[res.x, res.y, res.z, res.visibility]
                        for res in results.pose_landmarks.landmark]).flatten()
    except Exception as ex:
        # print("No Pose Landmarks")
        pose = np.zeros(33 * 4)

    try:
        left_hand = np.array([[res.x, res.y, res.z]
                              for res in results.left_hand_landmarks.landmark]).flatten()
    except Exception as ex:
        # print("No Left Hand Landmarks")
        left_hand = np.zeros(21*3)

    try:
        right_hand = np.array([[res.x, res.y, res.z]
                               for res in results.right_hand_landmarks.landmark]).flatten()
    except Exception as ex:
        # print("No Right Hand Landmarks")
        right_hand = np.zeros(21*3)

    keypoints = np.concatenate([face, pose, left_hand, right_hand])  # (1662,)

    return keypoints


def draw_landmarks(image, results):
    # Face Connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(
                                  color=(0, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(0, 0, 255), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(0, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(0, 0, 255), thickness=1, circle_radius=1)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(0, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(0, 0, 255), thickness=1, circle_radius=1)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(0, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(0, 0, 255), thickness=1, circle_radius=1)
                              )


def frames_extraction(video_path, holistic, frames_per_video, create_sample):
    frames_list = []
    keypoints_list = []
    # print(f"Video Path: {video_path}")
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_frames_window = max(int(video_frames_count/frames_per_video), 1)

    if create_sample:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('output.mp4', fourcc, 10.0,
                              (IMAGE_HEIGHT, IMAGE_WIDTH))

    for frame_counter in range(frames_per_video):

        video_reader.set(cv2.CAP_PROP_POS_FRAMES,
                         frame_counter * skip_frames_window)
        success, frame = video_reader.read()

        if not success:
            print(f"Not Successful: {video_path}")
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        resized_frame, results = mediapipe_detection(resized_frame, holistic)

        keypoints = extract_keypoints(results)  # (1662,)

        draw_landmarks(resized_frame, results)
        # print(results.pose_landmarks)
        # print("show")
        # color_map = plt.imshow(resized_frame)
        # plt.savefig("out.png")

        normalized_frame = resized_frame / 255

        frames_list.append(normalized_frame)
        keypoints_list.append(keypoints)  # (20, 1662)

        if create_sample:
            out.write(resized_frame)

    video_reader.release()
    return frames_list, keypoints_list


def create_dataset(root_dir, frames_per_video, create_sample):

    features_frames = []
    labels = []

    features_keypoints = []

    # action_dir = "UCF-101"
    label_encoder = dict()
    class_index = 0

    actions = os.listdir(root_dir)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for index, action in enumerate(actions):

            for sub_action in os.listdir(os.path.join(root_dir, action)):

                for video in os.listdir(os.path.join(root_dir, action, sub_action)):

                    video_path = os.path.join(
                        root_dir, action, sub_action, video)

                    frames, keypoints = frames_extraction(
                        video_path, holistic, frames_per_video, create_sample)

                    if len(frames) == frames_per_video:
                        features_frames.append(frames)
                        labels.append(index)
                        features_keypoints.append(keypoints)

                        if action not in label_encoder:
                            label_encoder[action] = class_index
                            class_index += 1
                            print(
                                f'Extracting Data of Class: {action}')

                    if create_sample:
                        break
                if create_sample:
                    break
            if create_sample:
                break

    features_frames = np.asarray(features_frames)
    labels = np.array(labels)
    features_keypoints = np.asarray(features_keypoints)

    if not create_sample:
        features = np.moveaxis(features_frames, 4, 1)

    return features_frames, labels, features_keypoints, label_encoder


def save_files(split, temp, features_frames, labels, features_keypoints):
    print(f"Full Dataset {temp}")
    print(features_frames.shape, labels.shape, features_keypoints.shape)

    assert features_frames.shape[0] == features_keypoints.shape[0], "Save problem with shape"
    assert labels.shape[0] == features_keypoints.shape[0], "Save problem with shape"
    np.save(os.path.join(
        split, f"X_{temp}_frames.npy"), features_frames, allow_pickle=True)

    np.save(os.path.join(
        split, f"y_{temp}.npy"), labels, allow_pickle=True)

    np.save(os.path.join(
        split, f"X_{temp}_keypoints.npy"), features_keypoints, allow_pickle=True)


def save(model, features_frames, labels_frames, features_keypoints):
    train_indexes, val_indexes, test_indexes = get_splits(
        features_frames, features_keypoints, labels)

    print("Full Dataset")
    print(features_frames.shape, labels.shape, features_keypoints.shape)

    X_train_frames = features_frames[train_indexes]
    y_train = labels[train_indexes]
    X_train_keypoints = features_keypoints[train_indexes]

    X_test_frames = features_frames[test_indexes]
    y_test = labels[test_indexes]
    X_test_keypoints = features_keypoints[test_indexes]

    X_val_frames = features_frames[val_indexes]
    y_val = labels[val_indexes]
    X_val_keypoints = features_keypoints[val_indexes]

    save_files(f"{model}/train", "train", X_train_frames,
               y_train, X_train_keypoints)

    save_files(f"{model}/val", "val", X_test_frames, y_test, X_test_keypoints)

    save_files(f"{model}/test", "test", X_val_frames, y_val, X_val_keypoints)


def get_splits(a, b, y):

    assert a.shape[0] == b.shape[0], "Problem with shape a, b"
    assert y.shape[0] == b.shape[0], "Problem with shape y, b"

    indexes = np.arange(0, a.shape[0])
    random.shuffle(indexes)

    test_size = int(len(indexes) * 0.2)

    test_indexes = indexes[:test_size]

    indexes = indexes[test_size:]
    random.shuffle(indexes)
    val_size = int(len(indexes) * 0.2)

    val_indexes = indexes[:val_size]

    train_indexes = indexes[val_size:]

    assert len(train_indexes) > len(val_indexes) + \
        len(test_indexes), "Problem with length"

    assert len(train_indexes) + len(val_indexes) + \
        len(test_indexes) == a.shape[0], "Problem with total length"
    return train_indexes, val_indexes, test_indexes


if __name__ == "__main__":

    frames_per_video = 20

    create_sample = False

    print(f"create_sample: {create_sample}")

    features_frames, labels, features_keypoints, label_encoder = create_dataset(
        "UCF-11", frames_per_video, create_sample)

    if not create_sample:

        with open("label_encoder.json", 'w') as fp:
            json.dump(label_encoder, fp, indent=4)

        save("Conv-LSTM", features_frames, labels, features_keypoints)
