import torch
import cv2
import numpy as np
from constants import classes

path_to_model = "mvit32-2.pt"
path_to_output_file = "subtitles.txt"

model = torch.jit.load(path_to_model)
model.eval()
window_size = 16  # from model name
threshold = 0.5
frame_interval = 1
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]


def resize(im, new_shape=(224, 224)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
    return im


def video_process(path_to_input_video):
    cap = cv2.VideoCapture(path_to_input_video)
    _, frame = cap.read()
    shape = frame.shape

    tensors_list = []
    prediction_list = ["---"]
    output_text = ""

    frame_counter = 0
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        frame_counter += 1
        if frame_counter == frame_interval:
            image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            image = resize(image, (224, 224))
            image = (image - mean) / std
            image = np.transpose(image, [2, 0, 1])
            tensors_list.append(image)
            if len(tensors_list) == window_size:
                input_tensor = np.stack(tensors_list[: window_size], axis=1)[None][None]
                input_tensor = input_tensor.astype(np.float32)
                input_tensor = torch.from_numpy(input_tensor)
                with torch.no_grad():
                    outputs = model(input_tensor)[0]
                gloss = str(classes[outputs.argmax().item()])
                if outputs.max() > threshold:
                    if gloss != prediction_list[-1] and len(prediction_list):
                        if gloss != "---":
                            prediction_list.append(gloss)
                tensors_list.clear()
            frame_counter = 0

        text = "  ".join(prediction_list)
        output_text = text
    cap.release()
    return output_text

