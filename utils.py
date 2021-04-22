import matplotlib.pyplot as plt
from torch import Tensor
from torchvision.transforms import ToPILImage
from PIL import ImageDraw


def get_coco_classes():
    classes = []
    with open('coco_classes.txt') as input_file:
        for line in input_file:
            classes.append(line.strip())
    return classes


def draw_bbox(pil_img, text, box: Tensor, color, frame_width=1):
    xy = (box[:2] + frame_width).tolist()
    img = ImageDraw.Draw(pil_img)
    text_size = img.textsize(text)
    text_bbox = xy + [xy[0] + text_size[0], xy[1] + text_size[1]]
    if (diff := text_bbox[2] - pil_img.size[0]) > 0:
        xy[0] -= diff
        text_bbox[0] -= diff
        text_bbox[2] -= diff
    color = tuple(int(i*255) for i in color)
    img.rectangle(box.tolist(), outline=color, width=frame_width)
    img.rectangle(text_bbox, fill=color)
    img.text(xy, text, fill=(255,255,255,255))


def draw_class_labels(dets, imgs, classes=None, threshold=0.6):
    results = []
    cmap = plt.get_cmap('tab20')
    for det, image in zip(dets, imgs):
        pil_img = ToPILImage()(image).convert('RGB')
        for box, score, label in zip(det['boxes'], det['scores'], det['labels']):
            if score.item() >= threshold:
                draw_label = label.item() if classes is None else classes[label.item()-1]
                text = str(draw_label) + ' ' + str(round(score.item(), 1))
                color = cmap(label.item()%20)[:3]
                draw_bbox(pil_img, text, box, color)
        results.append(pil_img)
    return results


def draw_object_id(dets, pil_img):
    cmap = plt.get_cmap('tab20')
    for det in dets:
        text = 'Object id: {}'.format(det[4])
        color = cmap(det[4]%20)[:3]
        draw_bbox(pil_img, text, Tensor(det[:4]), color)
    return pil_img
