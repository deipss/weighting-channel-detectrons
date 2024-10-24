import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
"""
播放一个视频
"""

def coco_analysis():
    val_coco = COCO('/data/ai_data/Brackish/annotations/annotations_COCO/train_groundtruth.json')
    print(val_coco.cats)
    ids = val_coco.catToImgs[2]
    ids_unique =list(set(ids))
    print(ids_unique)
    imgs_cate= [val_coco.imgs.get(e)for e in ids_unique]
    anno_cate = [val_coco.imgToAnns.get(e) for e in ids_unique]
    anno_cate_1d = [item for sublist in anno_cate for item in sublist]
    print(len(imgs_cate))
    print(len(anno_cate_1d))
    print(len(ids_unique))



def play_video(video_path):
    # 创建VideoCapture对象，传入视频文件路径
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 循环读取视频帧
    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 如果读取成功，ret为True
        if ret:
            # 显示帧
            cv2.imshow('Video', frame)

            # 按下q键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # 视频结束，退出循环
            break

    # 释放VideoCapture对象
    cap.release()

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()


def show_by_transfomer():
    img = cv2.imread(
        '/Users/deipss/workspace/ai/weighting-channel-detectrons/data/Brackish/dataset/img/2019-02-20_19-01-02to2019-02-20_19-01-13_1-0006.png')
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    cv2.imshow('img', img)
    num = transform(img).numpy().transpose(1, 2, 0)
    cv2.imshow('transform', num)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_json_of_ground_truth():
    # 路径设置
    annotation_file = '/Users/deipss/workspace/ai/weighting-channel-detectrons/data/Brackish/annotations/annotations_COCO/test_groundtruth.json'
    image_folder = '/Users/deipss/workspace/ai/weighting-channel-detectrons/data/Brackish/dataset/img'

    # 读取标注文件
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # 提取注释信息
    annotations = coco_data['annotations']
    images = coco_data['images']
    categories = coco_data['categories']

    # 创建类别 ID 到类名的映射
    category_id_to_name = {category['id']: category['name'] for category in categories}

    fig, axes = plt.subplots(nrows=6, ncols=2,figsize=(10,20))

    for idx, image_id in enumerate([1, 2, 33, 87, 23, 111]):

        # 查找对应的图像信息
        image_info = next(img for img in images if img['id'] == image_id)
        image_path = os.path.join(image_folder, image_info['file_name'])

        # 获取该图像的所有注释
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

        # 绘制边界框
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
        img = transform(image).numpy().transpose(1, 2, 0)
        draw = ImageDraw.Draw(image)

        # 遍历注释信息
        for annotation in annotations:
            if annotation['image_id'] == image_info['id']:
                bbox = annotation['bbox']
                category_id = annotation['category_id']
                category_name = category_id_to_name[category_id]

                # 绘制边界框
                x, y, width, height = bbox
                draw.rectangle([(x, y), (x + width, y + height)], outline='red', width=2)
                draw.text((x, y - 10), category_name, fill='red')

        ax = axes[idx, 0]
        ax_t = axes[idx, 1]

        # 显示图像
        ax.imshow(np.array(image))
        ax.set_title(str(image_id))
        ax.axis('off')

        ax_t.imshow(np.array(img))
        ax_t.set_title(str(image_id) + '_t')
        ax_t.axis('off')
    plt.show()


def load_helper_json():
    # 打开训练集中的帮助文件
    with  open(
            '/Users/deipss/workspace/ai/weighting-channel-detectrons/data/Brackish/annotations/annotations_COCO/train_helper_dirs.json',
            'r') as f:
        data = json.load(f)
        pass
    # 打开训练集中的详情文件
    with  open(
            '/Users/deipss/workspace/ai/weighting-channel-detectrons/data/Brackish/annotations/annotations_COCO/train_groundtruth.json',
            'r') as f:
        data = json.load(f)
        pass


if __name__ == '__main__':
    coco_analysis()




class BrackishDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Convert the image to a PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.images = self.annotations['images']
        self.annotations = self.annotations['annotations']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        annotations = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]
        labels = [ann['category_id'] for ann in annotations]
        bboxes = [ann['bbox'] for ann in annotations]

        if self.transform:
            image = self.transform(image)

        return image, labels, bboxes
