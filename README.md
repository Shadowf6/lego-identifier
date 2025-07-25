### Lego Identifier

This program takes in an image of a lego piece, and outputs the name and number of the lego piece.

[image description, direct image link]

## Algorithm

The program uses a pretrained resnet-18 model that is trained with transfer learning on a new dataset consisting of legos. The program takes in an input image, which is then processed using Imagenet from the Jetson Inference library. Imagenet is a resnet-18 model trained on the ILSVRC, a large dataset of 100 objects. The program processes the image and outputs a guess on what lego the input image is.

## Running the Project

You will need: Jetson Nano

1. Install the Jetson Inference library onto your Nano. The library also installs torch, imagenet, and the resnet18 model, libraries the project will be using.
2. SSH into the Nano.
3. Clone the project repository onto the Nano using the command `git clone https://github.com/Shadowf6/nvidia-jetson-ai`
4. Run the command `cd jetson-inference`
5. Pick a random number between 1 and 100 and type the command `FILE=[num].png` (or upload your own image onto the nano and add the file name)
6. Run the command imagenet --model=model.onnx --labels=data/labels.txt --input-blob=input_0--output-blob=ouput_0 $FILE
7. The output should include a guess and a confidence level on which lego the image contains.
