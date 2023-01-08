import argparse
import os
import time
import warnings

import cv2
import numpy as np

IMAGE = ['.jpg', '.png']
from torchvision import transforms
import json
from PIL import Image
import torch
from efficientnet_pytorch import EfficientNet
def multi_images_test(img_path, model_name, weights_path):
    t1 = time.time()
    model = EfficientNet.from_pretrained(model_name, weights_path=weights_path, num_classes=3)

    # tfms图像预处理 把图片resize 以及 转化成tensor
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img_paths = os.listdir(img_path)
    img = []
    for path in img_paths:
        if path[-4:] in IMAGE:
            # im = Image.open(os.path.join(img_path, path)).convert('RGB')
            im = cv2.imread((os.path.join(img_path, path)), 1)
            im = Image.fromarray(im)
            img.append(tfms(im).numpy())
    img = torch.tensor(img)
    print(img.shape)

    # Load ImageNet class names
    labels_map = json.load(open('./datasets/luowenhuajian/luowen.txt'))
    labels_map = [labels_map[str(i)] for i in range(3)]
    # Classify
    model.eval()
    print('-------------------------------')
    with torch.no_grad():
        outputs = model(img)
    for i, pred in enumerate(outputs):
        for idx in torch.topk(pred, k=1).indices.tolist():
            prob = torch.softmax(pred, dim=0)[idx].item()
            print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob * 100))
            print(img_paths[i])
    print('--------------------------------')
    t2 = time.time()
    print("*********************************************")
    print(f"{model_name} takes time:{t2 - t1}s")
    print("*********************************************")
"""
@Param: 
source 图像源所在文件夹

@:return
img_dic:    图像名list
imgs: tensor格式的图像
"""
def load_images(source):
    # tfms图像预处理 把图片resize 以及 转化成tensor
    # Preprocess image
    img_names = []
    tfms = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_paths = os.listdir(source)
    img = []
    for path in img_paths:
        if path[-4:] in IMAGE:
            # im = Image.open(os.path.join(img_path, path)).convert('RGB')
            im = cv2.imread((os.path.join(source, path)), 1)
            im = Image.fromarray(im)
            img.append(tfms(im).numpy())
            img_names.append(path)
    img = torch.tensor(img)
    img_names = sorted(img_names)
    return img_names, img

"""
将两个图片拼接并保存
@:param
image0  图像0
image1  图像1
imagePath  拼接保存的图片
"""
def saveHconcatImg(image0, image1, imagePath):
    mid = np.zeros((image0.shape[0], 5, 3), dtype=np.uint8)
    mid[:, :] = [0, 255, 0]
    concat_image = cv2.hconcat([image0, mid, image1])
    cv2.imwrite(imagePath, concat_image)
"""
@Param:
weights: 权重文件路径
source1: teaching示教图
source2: sample测试图
result_json  保存结果json文件的txt文本路径
result_imgs  保存不合格图像的结果路径
"""
def folderQualityInspection(weights="./checkpoints/efficientnet-b3_18_46_15.pth",
                            source1='./datasets/luowenhuajian/test/teaching/',
                            source2='./datasets/luowenhuajian/test/sample/',
                            result_json='./datasets/luowenhuajian/test/result_json.txt',
                            result_imgs='./datasets/luowenhuajian/test//result_img_folder/',
                            num_class=3):
    # 判断路径不存在则创建
    if not os.path.exists(result_imgs):
        os.makedirs(result_imgs)
    model_name = 'efficientnet-b3'
    t1 = time.time()
    model = EfficientNet.from_pretrained(model_name, weights_path=weights, num_classes=num_class)

    # 加载图片
    img_names1, imgs1 = load_images(source1)
    img_names2, imgs2 = load_images(source2)

    # Load ImageNet class names

    # labels_map = json.load(open('./datasets/luowenhuajian/luowen.txt'))
    # labels_map = [labels_map[str(i)] for i in range(3)]
    # Classify
    model.eval()
    print('-------------------------------')
    with torch.no_grad():
        outputs1 = model(imgs1)
        outputs2 = model(imgs2)
    _, preds1 = torch.max(outputs1, 1)
    _, preds2 = torch.max(outputs2, 1)
    preds1 = preds1.numpy()
    preds2 = preds2.numpy()

    index = np.arange(len(preds1))
    # 返回预测结果不相等所在的索引值
    print(f"index:{index[preds1 != preds2]}")
    index = index[preds1 != preds2]

    qualified = True
    unqualified_image_path = []
    if len(index) == 0:
        # 合格
        print("qualified: true")
    else:
        # 不合格
        qualified = False
        for idx in index:
            img1 = cv2.imread(os.path.join(source1, img_names1[idx]), 1)
            img2 = cv2.imread(os.path.join(source2, img_names2[idx]), 1)
            unqualified_image_path.append(os.path.join(result_imgs, img_names2[idx]))
            saveHconcatImg(img1, img2, os.path.join(result_imgs, img_names2[idx]))

    # 不合格json
    result = {"UnqualifiedImagePath": unqualified_image_path, 'qualified': qualified}
    with open(result_json, 'w', encoding='utf-8') as file_obj:
        json.dump(result, file_obj, indent=2)
    print('--------------------------------')
    t2 = time.time()
    print("*********************************************")
    print(f"{model_name} takes time:{t2 - t1}s")
    print("*********************************************")


"""
可选参数
"""
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./checkpoints/luowenHuajianCheck.pth', help='model path(s)')
    parser.add_argument('--source1', type=str, default='./datasets/luowenhuajian/test/teaching/')
    parser.add_argument('--source2', type=str, default='./datasets/luowenhuajian/test/sample/')
    parser.add_argument('--result_json', type=str, default='./datasets/luowenhuajian/test/result_json.txt')
    parser.add_argument('--result_imgs', type=str, default='./datasets/luowenhuajian/test/result_img_folder/')
    opt = parser.parse_args()
    return opt


def main(opt):
    folderQualityInspection(**vars(opt))


if __name__ == "__main__":
    # multi_images_test(img_path='./datasets/luowenhuajian/rorated_val/1', model_name='efficientnet-b3', weights_path="./checkpoints/efficientnet-b3_18_46_15.pth")
    # multi_images_test(img_path='./datasets/luowenhuajian/rorated_val/2', model_name='efficientnet-b3', weights_path="./checkpoints/efficientnet-b3_18_46_15.pth")
    # multi_images_test(img_path='./datasets/luowenhuajian/rorated_val/3', model_name='efficientnet-b3', weights_path="./checkpoints/efficientnet-b3_18_46_15.pth")
    warnings.filterwarnings("ignore")
    opt = parse_opt()
    main(opt)