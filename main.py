import os
from argparse import ArgumentParser
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from image_detection import DetectImageDataset, detect
from video_detection import video_detect


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', nargs=1, required=True, help='input dir with images or path to video file')
    parser.add_argument('-t', action='store_true', help='enable object tracking using video input')
    args = parser.parse_args()
    input_path = args.i[0]
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    if not os.path.exists(input_path):
        print('Path does not exists.')
    elif os.path.isfile(input_path):
        if args.t:
            video_detect(model, input_path)
        else:
            video_detect(model, input_path, track=False)
    elif os.path.isdir(input_path) and not args.t:
        data = DetectImageDataset(input_path)
        detect(model, data)
    else:
        print('Incorrect arguments.')
