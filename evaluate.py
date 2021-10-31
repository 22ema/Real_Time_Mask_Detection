from inference.inference import *
import os

if __name__ == "__main__":
    dataset_path = "./media/dataset"
    annotation_path = "./media/dataset/annotation"
    weights = './model/yolor/epoch_249.pt'
    cfg = './Yolor/cfg/yolor_p6.cfg'
    test_image_list = os.listdir(dataset_path)

    model = setting_model(cfg, weights)
    for test_image in test_image_list[:10]:
        if '.jpg' in test_image:
            annotation_name = test_image[:-3]+'txt'
            image_path = os.path.join(dataset_path, test_image)
            annotation_path = os.path.join(annotation_path, annotation_name)
            image = cv2.imread(image_path)
            output, ori_image, trans_image = inference(image, model)
            for si, pred in enumerate(output):
                x = pred.clone()
                x[:, :4] = scale_coords(trans_image[si].shape[1:], x[:, :4], ori_image.shape)
                for *xyxy, conf, cls in x:

                    print(xyxy)
                    print(conf)
                    print(cls)
