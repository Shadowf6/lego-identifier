#!/usr/bin/python3

import jetson_inference
import jetson_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet("lego-identifier.onnx", labels="labels.txt", input_blob="input_0", output_blob="output_0")

idx, confidence = net.Classify(img)
desc = net.GetClassDesc(idx)

example_images = {
    0: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=2357#T=C",
    1: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3003#T=C",
    2: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3004#T=C",
    3: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3005#T=C",
    4: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3022#T=C",
    5: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3023#T=C",
    6: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3024#T=C",
    7: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3040#T=C",
    8: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3069#T=C",
    9: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3673#T=C",
    10: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3713#T=C",
    11: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3794#T=C",
    12: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=6632#T=C",
    13: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=11214#T=C",
    14: "https://www.bricklink.com/v2/catalog/catalogitem.page?P=18651#T=C",
    15: "https://www.brickowl.com/us/catalog/lego-half-bushing-32123-42136"
}

if (confidence > 0.5):
    print("\nLego type is " + str(desc) + " with confidence " + str(round(100*confidence,2)) + "%")
    print("This is a link of what type of lego the model thinks it is: " + example_images[idx])
else:
    print("\nThe model cannot identify what type of lego this is.")
