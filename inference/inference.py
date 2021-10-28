import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from Yolor.utils.general import non_max_suppression
from torch.utils.data import WeightedRandomSampler
from Yolor.models.models import *

def setting_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights = './model/yolor/best_overall.pt'
    cfg = './Yolor/cfg/yolor_p6.cfg'
    model = Darknet(cfg).to(device)
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model

def inference(ori_image, model):
    labels = ['mask_weared_incorrect', 'with_mask', 'without_mask']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    imgsz = check_img_size(640, s=64)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    res_image = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(res_image).float()
    image = image.permute(2,0,1).squeeze(0).unsqueeze(0)
    img = image.to(device, non_blocking=True)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # img = torch.zeros((1, 3, 640, 640), device=device)  # init img
    # img = img.float()  # uint8 to fp16/32
    with torch.no_grad():
        inf_out, train_out = model(img)
        output = non_max_suppression(inf_out, conf_thres=0.2, iou_thres=0.65, classes=[0,1,2])
        print("output:",output)
        for si, pred in enumerate(output):
            x = pred.clone()
            x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], ori_image.shape)  # to original
            c=x[:,-1].unique()
            n = (x[:, -1] == c).sum()
            print(c)
            for *xyxy, conf, cls in x:
                print(cls)
                print(conf)
                if int(cls) == 0:
                    cv2.putText(ori_image, labels[0], (xyxy[0]-10, xyxy[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0),1,cv2.LINE_AA)
                    result_image = cv2.rectangle(ori_image, (xyxy[0], xyxy[1]),(xyxy[2],xyxy[3]),(0,0,255),3)
                elif int(cls) == 1:
                    cv2.putText(ori_image, labels[1], (xyxy[0]-10, xyxy[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0),1,cv2.LINE_AA)
                    result_image = cv2.rectangle(ori_image, (xyxy[0], xyxy[1]),(xyxy[2],xyxy[3]),(0,255,0),3)
                elif int(cls) == 2:
                    cv2.putText(ori_image, labels[2], (xyxy[0]-10, xyxy[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0),1,cv2.LINE_AA)
                    result_image = cv2.rectangle(ori_image, (xyxy[0], xyxy[1]),(xyxy[2],xyxy[3]),(255,0,0),3)
                return result_image
            return ori_image



