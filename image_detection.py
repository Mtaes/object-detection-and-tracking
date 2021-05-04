import os
import pathlib
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from utils import draw_class_labels, get_coco_classes


class DetectImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir):
        super().__init__()
        self.input_dir = input_dir
        self.images = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.input_dir, self.images[idx])).convert('RGB')
        return to_tensor(img)
    
    def __len__(self):
        return len(self.images)


def detect(model, dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,  num_workers=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        model.eval()
        model.to(device)
        dets = [model(image.to(device))[0] for image in data_loader]
    results = draw_class_labels(dets, dataset, get_coco_classes())
    output_dir = os.path.join(dataset.input_dir, 'detections')
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    output_dir = os.path.join(output_dir, '{}.jpg')
    for img, i in zip(results, range(len(results))):
        img.save(output_dir.format(i))
