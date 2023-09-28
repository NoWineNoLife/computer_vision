# -*- coding = utf-8 -*-
#  * @file              : labelme2png.py
#  * @author            : tsing
#  * @brief             : None
#  * @attention         : None
#  * @time              : 2023/8/24 下午1:14
# @function: the script is used to do something.

import os
import json
import cv2
import yaml
import numpy as np
from glob import glob


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(current_path, "config", "kaggle_car.yaml")
    with open(yaml_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if "base" in config:
        base_yaml_file = os.path.join(current_path, "config", config["base"])
        with open(base_yaml_file, 'r') as f_base:
            base_config = yaml.load(f_base, Loader=yaml.FullLoader)
            del config["base"]
        base_config.update(config)
        config = base_config

    json_regex = config.get("dataset").get("json_regex")
    store_path = config.get("dataset").get("store_path")

    labels = config.get("labels")

    all_json_files = glob(json_regex)
    for json_name in all_json_files:
        json_file = open(json_name, 'r')
        json_data = json.loads(json_file.read())

        img_name = os.path.splitext(os.path.basename(json_name))[0]
        img_file = str(json_name).replace("json", "jpg")
        img = cv2.imread(img_file)
        img_h, img_w, _ = img.shape

        img_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        shapes = json_data["shapes"]
        for shape in shapes:
            contour = shape["points"]
            contour_converted = []
            for contour_points in contour:
                point = np.array(contour_points).astype(np.int32)
                contour_converted.append(point)

            contour_converted = np.array(contour_converted)

            try:
                idx = labels.index(shape["label"]) + 1
                cv2.fillPoly(img_mask, [contour_converted], idx)
            except:
                print("could not find label")

            base_name = os.path.basename(img_file)
            store_dir = os.path.join(store_path, base_name.replace("jpg", "png"))
            cv2.imwrite(store_dir, img_mask)












