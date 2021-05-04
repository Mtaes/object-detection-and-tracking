import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from utils import draw_object_id, get_coco_classes, draw_class_labels
from sort.sort import Sort


def video_detect(model, path_to_video, threshold=0.6, track=True):
    mot_tracker = Sort()
    cap = cv2.VideoCapture(path_to_video)
    out = cv2.VideoWriter(path_to_video + '-detections.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (640,  480))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        model.eval()
        model.to(device)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('No more frames')
                break
            pil_img = Image.fromarray(frame)
            tensor_img = to_tensor(pil_img).unsqueeze_(0)
            dets = model(tensor_img.to(device))
            if track:
                tracked_dets = None
                for box, score in zip(dets[0]['boxes'], dets[0]['scores']):
                    if score.item() >= threshold:
                        tracked_det = np.array([torch.cat((box, score.reshape(1))).detach().cpu().numpy()])
                        tracked_dets = np.concatenate((tracked_dets, tracked_det)) if tracked_dets is not None else tracked_det
                tracked_dets = mot_tracker.update(tracked_dets if tracked_dets is not None else np.empty((0, 5)))
                out.write(np.array(draw_object_id(tracked_dets, pil_img)))
            else:
                out.write(np.array(draw_class_labels(dets, tensor_img, get_coco_classes(), threshold=threshold)[0]))
    cap.release()
    out.release()
    cv2.destroyAllWindows()
