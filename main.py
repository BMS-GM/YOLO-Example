import cv2
import argparse
import numpy as np


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, colors, classes, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = colors[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_network(frame, width, height, scale, colors, classes, net):
    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(frame, colors, classes, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    #cv2.destroyAllWindows()
    #cv2.imshow("object detection", frame)

    return frame


def main():
    # Read CMD ARGs
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--video', required=True,
                    help='path to input image')
    ap.add_argument('-c', '--config', required=True,
                    help='path to yolo config file')
    ap.add_argument('-w', '--weights', required=True,
                    help='path to yolo pre-trained weights')
    ap.add_argument('-cl', '--classes', required=True,
                    help='path to text file containing class names')
    args = ap.parse_args()

    scale = 0.00392

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    # Run Network
    vid_capture = cv2.VideoCapture(args.video)
    length = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    mod_len = length//100
    success, frame = vid_capture.read()
    width = frame.shape[1]
    height = frame.shape[0]

    count = 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter("result.mp4v", fourcc, 20.0, (width, height))

    while success:
        if count % mod_len == 0:
            print("Processing Frames: %f%%" % ((count/length) * 100))

        out.write(run_network(frame, width, height, scale, colors, classes, net))

        success, frame = vid_capture.read()
        count += 1

    # Release everything if job is finished
    out.release()
    print("DONE!!")


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
