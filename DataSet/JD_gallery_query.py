from __future__ import absolute_import, print_function

"""
JD test
"""
import torch
import torch.utils.data as data
from PIL import Image
import os
from torchvision import transforms
from collections import defaultdict

# Solve IOError
from PIL import ImageFile
from torch.backends import cudnn

cudnn.benchmark = True

ImageFile.LOAD_TRUNCATED_IMAGES = True


def default_loader(path, area=None):
    if area is None:
        return Image.open(path).convert('RGB')
    else:
        return Image.open(path).crop(area).convert('RGB')


class JD_Data(data.Dataset):
    def __init__(self, imgs=None, labels=None, areas=None, loader=default_loader, transform=None):

        # Initialization data path and train(gallery or query) txt path

        self.images = imgs
        self.labels = labels
        self.areas = areas

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if transform is None:
            transform = transforms.Compose([
                # transforms.CovertBGR(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(scale=(0.16, 1), size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        # read txt get image path and labels

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        # Initialization Done
        self.classes = classes
        self.transform = transform
        self.Index = Index
        self.loader = loader

    def __getitem__(self, index):

        if self.areas is not None:
            fn, label, area = self.images[index], self.labels[index], self.areas[index]
            img = self.loader(fn, area=area)
        else:
            fn, label = self.images[index], self.labels[index]
            img = self.loader(fn)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


class JD_Gallery_Query:
    def __init__(self, root=None, transform=None, crop=True, origin_width=288, width=256):
        # Data loading code
        self.crop = crop

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if transform is None:
            transform = [

                transforms.Compose([
                    # transforms.CovertBGR(),
                    transforms.Resize(origin_width),
                    transforms.TenCrop(width, vertical_flip=False),
                    transforms.Lambda(
                        lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
                ]),

                transforms.Compose([
                    # transforms.CovertBGR(),
                    transforms.Resize(width),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ]),

                transforms.Compose([
                    # transforms.CovertBGR(),
                    transforms.Resize(width),
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])]

        if root is None:
            root = "/opt/intern/users/xunwang/jd-comp/"

        label_txt = os.path.join(root, 'q_all.txt')
        gallery_label_txt = "/opt/intern/users/xunwang/jd-comp/labels/jd-fashion-comp/fashion_retrieval/S.txt"
        query_dir = '/opt/intern/users/xunwang/jd-comp/images/Q'
        gallery_dir = '/opt/intern/users/xunwang/jd-comp/images/S'

        # read txt get image path and labels

        file = open(label_txt)
        images_anon = file.readlines()
        query_images = []
        query_labels = []
        query_areas = []
        for i, img_anon in enumerate(images_anon):
            img_anon = img_anon.replace('com/', ' ')
            img_anon = img_anon.split(' ')
            if i == 0:
                print(img_anon)
            img_1 = os.path.join(query_dir, '%04d.jpg' % i)
            area_1 = [int(img_anon[i]) for i in range(2, 6)]
            query_images.append(img_1)
            query_areas.append(area_1)
            query_labels.append(i)

        file = open(gallery_label_txt)
        images_anon = file.readlines()
        gallery_images = []
        gallery_labels = []
        gallery_areas = []
        for i, img_anon in enumerate(images_anon):
            img_anon = img_anon.replace('com/', ' ')
            img_anon = img_anon.split(' ')
            if i == 0:
                print('gallery')
                print(img_anon)

            img_1 = os.path.join(gallery_dir, img_anon[1])
            area_1 = [int(img_anon[i]) for i in range(2, 6)]
            gallery_images.append(img_1)
            gallery_areas.append(area_1)
            gallery_labels.append(i)

        # gallery_images = gallery_images[970 * 128:]
        # gallery_areas = gallery_areas[970 * 128:]

        if self.crop:
            self.query = JD_Data(imgs=query_images, labels=query_labels, areas=query_areas, transform=transform[1])
            self.flip_query = JD_Data(imgs=query_images, labels=query_labels, areas=query_areas, transform=transform[2])
            self.gallery =\
                JD_Data(imgs=gallery_images, labels=gallery_labels, areas=gallery_areas, transform=transform[1])
            self.flip_gallery = \
                JD_Data(imgs=gallery_images, labels=gallery_labels, areas=gallery_areas, transform=transform[2])


def testJD_Fashion():
    data = JD_Gallery_Query(crop=True)
    img_loader = torch.utils.data.DataLoader(
        data.query, batch_size=10, shuffle=False, num_workers=2)
    for index, batch in enumerate(img_loader):
        print(batch[1])
        print(index)
        break
    print(len(data.query))


if __name__ == "__main__":
    testJD_Fashion()
