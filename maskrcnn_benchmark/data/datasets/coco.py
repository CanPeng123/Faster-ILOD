import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


# first 40 categories: 1 ~ 44; first 70 categories: 1 ~ 79; first 75 categories: 1 ~ 85
# second 40 categories: 45 ~ 91; second 10 categories: 80 ~ 91; second 5 categories: 86 ~ 91
# totally 80 categories
NUM_OLD_CATEGORY = 70  # do not include background
NUM_NEW_CATEGORY = 10  # number of added categories


COCO_VOC_CATS = ['__background__', 'airplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
                 'dog', 'horse', 'motorcycle', 'person', 'potted plant',
                 'sheep', 'couch', 'train', 'tv']

COCO_NONVOC_CATS = ['apple', 'backpack', 'banana', 'baseball bat',
                    'baseball glove', 'bear', 'bed', 'bench', 'book', 'bowl',
                    'broccoli', 'cake', 'carrot', 'cell phone', 'clock', 'cup',
                    'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee',
                    'giraffe', 'hair drier', 'handbag', 'hot dog', 'keyboard',
                    'kite', 'knife', 'laptop', 'microwave', 'mouse', 'orange',
                    'oven', 'parking meter', 'pizza', 'refrigerator', 'remote',
                    'sandwich', 'scissors', 'sink', 'skateboard', 'skis',
                    'snowboard', 'spoon', 'sports ball', 'stop sign',
                    'suitcase', 'surfboard', 'teddy bear', 'tennis racket',
                    'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light',
                    'truck', 'umbrella', 'vase', 'wine glass', 'zebra']

# COCO_CATS = COCO_VOC_CATS+COCO_NONVOC_CATS
COCO_CATS = ['__background__', 'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear',
             'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car',
             'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut',
             'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog',
             'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven',
             'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors',
             'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase',
             'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light',
             'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

coco_ids = {'airplane': 5, 'apple': 53, 'backpack': 27, 'banana': 52,
            'baseball bat': 39, 'baseball glove': 40, 'bear': 23, 'bed': 65,
            'bench': 15, 'bicycle': 2, 'bird': 16, 'boat': 9, 'book': 84,
            'bottle': 44, 'bowl': 51, 'broccoli': 56, 'bus': 6, 'cake': 61,
            'car': 3, 'carrot': 57, 'cat': 17, 'cell phone': 77, 'chair': 62,
            'clock': 85, 'couch': 63, 'cow': 21, 'cup': 47, 'dining table':
            67, 'dog': 18, 'donut': 60, 'elephant': 22, 'fire hydrant': 11,
            'fork': 48, 'frisbee': 34, 'giraffe': 25, 'hair drier': 89,
            'handbag': 31, 'horse': 19, 'hot dog': 58, 'keyboard': 76, 'kite':
            38, 'knife': 49, 'laptop': 73, 'microwave': 78, 'motorcycle': 4,
            'mouse': 74, 'orange': 55, 'oven': 79, 'parking meter': 14,
            'person': 1, 'pizza': 59, 'potted plant': 64, 'refrigerator': 82,
            'remote': 75, 'sandwich': 54, 'scissors': 87, 'sheep': 20, 'sink':
            81, 'skateboard': 41, 'skis': 35, 'snowboard': 36, 'spoon': 50,
            'sports ball': 37, 'stop sign': 13, 'suitcase': 33, 'surfboard':
            42, 'teddy bear': 88, 'tennis racket': 43, 'tie': 32, 'toaster':
            80, 'toilet': 70, 'toothbrush': 90, 'traffic light': 10, 'train':
            7, 'truck': 8, 'tv': 72, 'umbrella': 28, 'vase': 86, 'wine glass':
            46, 'zebra': 24}

coco_ids_to_cats = dict(map(reversed, list(coco_ids.items())))

min_keypoints_per_image = 10


def dict_slice(adict, start, end):
    keys = list(adict.keys())
    # print('keys : {0}'.format(keys))
    # print('length of keys: {0}'.format(len(keys)))
    dict_slice = {}
    for k in keys[start: end]:
        dict_slice[k] = adict[k]
    # print('dict_slice: {0}'.format(dict_slice))
    return dict_slice


def convert_cats_from_original_order_to_alphabetical_order(index):
    new_index = -1
    cats_to_ids = dict(map(reversed, enumerate(COCO_CATS)))
    coco_ids_to_alphabetical = {k: cats_to_ids[v] for k, v in coco_ids_to_cats.items()}
    for key in coco_ids_to_alphabetical:
        if key == index:
            new_index = coco_ids_to_alphabetical[key]
            break
    if new_index == -1:
        raise ValueError('Something is wrong!')
    return new_index


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


def train_class_data_check(anno):
    """
    only new categories' data
    """
    class_flag = False
    train_data_cats_dict = dict_slice(coco_ids_to_cats, NUM_OLD_CATEGORY, (NUM_OLD_CATEGORY + NUM_NEW_CATEGORY))
    # print('train_data_cats_dict: {0}'.format(train_data_cats_dict))
    train_data_cats_index = list(train_data_cats_dict.keys())  # index range is 0 ~ 90
    # print('train_data_cats_dict: {0}'.format(train_data_cats_dict))
    # print('length of train_data_cats_dict: {0}'.format(len(train_data_cats_dict)))
    for obj in anno:
        for train_index in train_data_cats_index:
            if obj['category_id'] == train_index:
                # print('category_id: {0}'.format(obj['category_id']))
                class_flag = True
                return class_flag
    return class_flag


def train_image_annotation(anno):
    """
    only new categories' annotations
    """
    train_data_cats_dict = dict_slice(coco_ids_to_cats, NUM_OLD_CATEGORY, (NUM_OLD_CATEGORY + NUM_NEW_CATEGORY))
    train_data_cats_index = list(train_data_cats_dict.keys())  # index range is 0 ~ 90
    new_anno = []
    for obj in anno:
        for train_index in train_data_cats_index:
            if obj["category_id"] == train_index:
                new_anno.append(obj)
                break
    return new_anno


def test_class_data_check(anno):
    """
    both old and new categories' data
    """
    class_flag = False
    test_data_cats_dict = dict_slice(coco_ids_to_cats, 0, (NUM_OLD_CATEGORY + NUM_NEW_CATEGORY))
    test_data_cats_index = list(test_data_cats_dict.keys())  # index range 0 ~ 90
    # print('test_data_cats_dict: {0}'.format(test_data_cats_dict))
    # print('length of test_data_cats_dict: {0}'.format(len(test_data_cats_dict)))
    for obj in anno:
        for test_index in test_data_cats_index:
            if obj['category_id'] == test_index:
                # print('category_id: {0}'.format(obj['category_id']))
                class_flag = True
                return class_flag
    return class_flag


def test_image_annotation(anno):
    """
    both old and new categories' annotations
    """
    test_data_cats_dict = dict_slice(coco_ids_to_cats, 0, (NUM_OLD_CATEGORY + NUM_NEW_CATEGORY))
    test_data_cats_index = list(test_data_cats_dict.keys())  # index range is 0 ~ 90
    new_anno = []
    for obj in anno:
        for test_index in test_data_cats_index:
            if obj["category_id"] == test_index:
                new_anno.append(obj)
                break
    return new_anno


class COCODataset(torchvision.datasets.coco.CocoDetection):

    def __init__(self, ann_file, root, remove_images_without_annotations, transforms=None, is_train=True):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.is_train = is_train
        count = 0

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                if self.is_train:
                    if train_class_data_check(anno):  # filtering images for new categories
                        count = count + 1
                        ids.append(img_id)
                else:
                    if test_class_data_check(anno):  # filtering images for old and new categories
                        count = count + 1
                        ids.append(img_id)
        self.ids = ids
        if self.is_train:
            print('number of images used for training: {0}'.format(count))
        else:
            print('number of images used for testing: {0}'.format(count))
        self.num_img = count

        self.json_category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        # filter annotation for old, new and exclude classes data
        if self.is_train:
            # print('before filtering, annotation: {0}'.format(anno))
            anno = train_image_annotation(anno)
            # print('after filtering, annotation: {0}'.format(anno))
        else:
            anno = test_image_annotation(anno)

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        # classes = [self.json_category_id_to_contiguous_id[c] for c in classes]  # convert index range from (0 ~ 90) to (0 ~ 80)
        # rearrange categories ids to alphabetical order
        # print('before alphabetical rearranging, classes: {0}'.format(classes))
        new_classes = []
        for cls in classes:
            new_cls = convert_cats_from_original_order_to_alphabetical_order(cls)
            new_classes.append(new_cls)
        # print('after alphabetical rearranging, classes: {0}'.format(new_classes))
        classes = new_classes
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            proposal = None
            img, target, proposal = self._transforms(img, target, proposal)

        return img, target, proposal, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def get_img_ids(self):
        if self.is_train:
            print('number of images used for training: {0}'.format(self.num_img))
        else:
            print('number of images used for testing: {0}'.format(self.num_img))

        return self.ids

    def get_included_cats(self):
        if self.is_train:
            data_cats_dict = dict_slice(coco_ids_to_cats, NUM_OLD_CATEGORY, (NUM_OLD_CATEGORY + NUM_NEW_CATEGORY))
        else:
            data_cats_dict = dict_slice(coco_ids_to_cats, 0, (NUM_OLD_CATEGORY + NUM_NEW_CATEGORY))
        data_cats_index = list(data_cats_dict.keys())  # index range 0 ~ 90
        print('coco.py | data_cats_index: {0}'.format(data_cats_index))
        data_cats_index.sort()
        print('coco.py | data_cats_index: {0}'.format(data_cats_index))
        print('coco.py | length of data_cats_index: {0}'.format(len(data_cats_index)))

        return data_cats_index


if __name__ == '__main__':

    cats_to_ids = dict(map(reversed, enumerate(COCO_CATS)))
    ids_to_cats = dict(enumerate(COCO_CATS))
    num_classes = len(COCO_CATS)
    categories = COCO_CATS[1:]

    print("coco_ids: {0}".format(coco_ids))
    print("coco_ids_to_cats: {0}".format(coco_ids_to_cats))
    cut_dict = dict_slice(coco_ids_to_cats, 70, 80)
    length_cut_dict = len(cut_dict)
    print("cut_dict: {0}".format(cut_dict))
    print("length_cut_dict: {0}".format(length_cut_dict))

    coco_ids_to_internal = {k: cats_to_ids[v] for k, v in coco_ids_to_cats.items()}
    ids_to_coco_ids = dict(map(reversed, coco_ids_to_internal.items()))
    print('coco_ids_to_internal: {0}'.format(coco_ids_to_internal))
    print('ids_to_coco_ids: {0}'.format(ids_to_coco_ids))

    new_index = convert_cats_from_original_order_to_alphabetical_order(74)
    print('new_index: {0}'.format(new_index))