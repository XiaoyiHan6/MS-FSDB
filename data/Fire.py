import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET

VISIFIRE_VOC_ROOT = "/data/1_VisiFire"
FIRESENSE_VOC_ROOT = "/data/2_FIRESENSE"
FURG_FIRE_DATASET_VOC_ROOT = "/data/3_furg-fire-dataset-master"
BOWFIREDATASET_VOC_ROOT = "/data/4_BoWFireDataset"
FIRE_SMOKE_DATASET_VOC_ROOT = "/data/5_FIRE-SMOKE-DATASET"
MY_FIRE_SMOKE_DATASET_ROOT = '/data/FireDataset'
TEST_ROOT = '/data/test'


def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes


class FireDetection(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__ ",
        "fire",
        "smoke")

    def __init__(self, root_dir, resize=[800, 1333], split='trainval', use_difficult=False, is_train=True,
                 augment=None):
        self.root = root_dir
        self.use_difficult = use_difficult
        self.imgset = split

        self._annopath = os.path.join(self.root, "labels", "%s.xml")
        self._imgpath = os.path.join(self.root, "images", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "layout", "%s.txt")

        with open(self._imgsetpath % self.imgset) as f:
            self.img_ids = f.readlines()
        self.img_ids = [x.strip() for x in self.img_ids]
        self.name2id = dict(zip(FireDetection.CLASSES_NAME, range(len(FireDetection.CLASSES_NAME))))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.resize = resize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.train = is_train
        self.augment = augment
        print("INFO=====>voc dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):

        img_id = self.img_ids[index]
        img = Image.open(self._imgpath % img_id)
        boxes = []
        classes = []
        if not os.path.exists(self._annopath % img_id):
            boxes = [[0., 0., 0., 0.]]
            classes = [-1]
        else:
            anno = ET.parse(self._annopath % img_id).getroot()

            for obj in anno.iter("object"):
                difficult = int(obj.find("difficult").text) == 1
                if not self.use_difficult and difficult:
                    continue
                _box = obj.find("bndbox")
                # Make pixel indexes 0-based
                # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
                box = [
                    _box.find("xmin").text,
                    _box.find("ymin").text,
                    _box.find("xmax").text,
                    _box.find("ymax").text,
                ]
                TO_REMOVE = 1
                box = tuple(
                    map(lambda x: x - TO_REMOVE, list(map(float, box)))
                )
                boxes.append(box)

                name = obj.find("name").text.lower().strip()
                classes.append(self.name2id[name])

        boxes = np.array(boxes, dtype=np.float32)
        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            if self.augment is not None:
                img, boxes = self.augment(img, boxes)
        img = np.array(img)
        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize)


        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return img, boxes, classes

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded, boxes
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num: max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes


if __name__ == "__main__":
    # def __call__(self,sample,resize=[400,666]) size:416
    # def __call__(self, sample, resize=[800, 1333]) size:832
    eval_dataset = FireDetection(root_dir=VISIFIRE_VOC_ROOT, resize=[400, 666],
                                 split='train', use_difficult=False, is_train=False, augment=None)
    img, boxes, classes = eval_dataset[0]
    print("boxes:", boxes)
    print("classes:", classes)
    # dataset=VOCDataset("/home/data/voc2007_2012/VOCdevkit/VOC2012",split='trainval')
    # for i in range(100):
    #     img,boxes,classes=dataset[i]
    #     img,boxes,classes=img.numpy().astype(np.uint8),boxes.numpy(),classes.numpy()
    #     img=np.transpose(img,(1,2,0))
    #     print(img.shape)
    #     print(boxes)
    #     print(classes)
    #     for box in boxes:
    #         pt1=(int(box[0]),int(box[1]))
    #         pt2=(int(box[2]),int(box[3]))
    #         img=cv2.rectangle(img,pt1,pt2,[0,255,0],3)
    #     cv2.imshow("test",img)
    #     if cv2.waitKey(0)==27:
    #         break
    # imgs,boxes,classes=eval_dataset.collate_fn([dataset[105],dataset[101],dataset[200]])
    # print(boxes,classes,"\n",imgs.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,imgs.dtype)
    # for index,i in enumerate(imgs):
    #     i=i.numpy().astype(np.uint8)
    #     i=np.transpose(i,(1,2,0))
    #     i=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
    #     print(i.shape,type(i))
    #     cv2.imwrite(str(index)+".jpg",i)
