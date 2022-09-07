import os
import re

import cv2
import pathlib
import pybboxes
import matplotlib.pyplot


def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)), ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]


def main():
    list_imgs = list([img for img in pathlib.Path("data/artefato/valid").rglob("*")])
    list_bounding_box_true = list([img for img in pathlib.Path("data/artefato/bounding_box_true").rglob("*")])
    list_results = list([result for result in pathlib.Path("results/artefato").rglob("*.txt")])
    print(len(list_imgs), len(list_results))

    # colors = {
    #     "manekia": (0, 153, 41),
    #     "ottonia": (153, 0, 112),
    #     "peperomia": (171, 47, 177),
    #     "piper": (245, 4, 0),
    #     "pothomorphe": (0, 241, 245)
    # }

    # cod_barra
    colors = {
        "barradeescala": (0, 153, 41),
        "carimbo": (153, 0, 112),
        "paletadecor": (171, 47, 177),
        "rotulo": (245, 4, 0),
        "envelope": (0, 241, 245),
        "codbarra": (253, 54, 126)
    }

    predictions = list([])

    for result in list_results:
        if not "especime" in str(result.stem):
            try:
                with open(result) as file:
                    lines = file.readlines()
                    file.close()
            except:
                pass
            _, _, _, label = re.split("[_]", str(result.stem))

            predictions_per_file = list()
            for l in lines:
                predictions_per_file.append(l.replace("\n", "") + f" {label.lower()}")

            predictions = predictions + predictions_per_file

    print(len(predictions))

    for img in list_imgs:
        pred_per_image = list(filter(lambda x: str(img.stem) in x, predictions))
        print(img.stem, len(pred_per_image))
        image = cv2.imread(str(img))
        image_threshold = cv2.imread(str(img))

        draw_bounding_box(colors, image, pred_per_image)
        draw_bounding_box(colors, image_threshold, pred_per_image, threshold=0.8)


def draw_bounding_box(colors, image, pred_per_image, threshold=0):
    color_bounding_box_true = (0, 0, 0)

    for p in pred_per_image:
        filename, confidence, x_min, y_min, width, height, label = re.split("[ ]", p)

        confidence = float(confidence)
        if confidence > threshold:
            x = float(x_min)
            y = float(y_min)
            w = float(width)
            h = float(height)

            dh, dw, _ = image.shape

            x_center, y_center, width, height = coco_to_yolo(x, y, w, h, dw, dh)
            print(x_center, y_center, width, height, label)

            x1 = int((x_center - width / 2) * dw)
            x2 = int((x_center + width / 2) * dw)
            y1 = int((y_center - height / 2) * dh)
            y2 = int((y_center + height / 2) * dh)

            text = f"{label}+{confidence}"
            image = cv2.rectangle(image, (x1, y1), (x2, y2), colors[label], 10)
            image = cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 3, color_bounding_box_true, 4)


            with open(os.path.join("data/artefato/bounding_box_true", f"{filename}.txt")) as file:
                lines = file.readlines()
                file.close()

            for l in lines:
                id, x_center, y_center, width, height = re.split("[ ]", l)
                if int(id) != 0:
                    print(x_center, y_center, width, height, label)

                    x1 = int((float(x_center) - float(width) / 2) * dw)
                    x2 = int((float(x_center) + float(width) / 2) * dw)
                    y1 = int((float(y_center) - float(height) / 2) * dh)
                    y2 = int((float(y_center) + float(height) / 2) * dh)

                    # labels = list(["manekia", "ottonia", "peperomia", "piper", "pothomorphe"])
                    # text = list(filter(lambda x: x in filename, labels))[0]
                    # text = text + "_bounding_box_true"
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 10)
                    # image = cv2.putText(image, text, (x2-10, y2), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)


    path_outfile = os.path.join("out/coco_artefato")
    pathlib.Path(path_outfile).mkdir(parents=True, exist_ok=True)
    if threshold != 0:
        path_outfile = os.path.join(path_outfile, f"{filename}_bb_threshold.png")
    else:
        path_outfile = os.path.join(path_outfile, f"{filename}_bb.png")
    cv2.imwrite(path_outfile, image)


if __name__ == '__main__':
    main()