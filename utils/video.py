import os

import cv2


def get_cap(path):
    if os.path.isdir(path):
        return FolderVideoReader(path)
    else:
        return cv2.VideoCapture(path)


def is_image(name):
    return '.jpg' in name or '.png' in name or '.bmp' in name


class FolderVideoReader:
    def __init__(self, path):
        self.dir = path
        self.img_list = [img for img in os.listdir(path) if 'mask' not in img and is_image(img)]
        self.img_list = sorted(self.img_list)
        self.i = 0
        self.opened = True

    def read(self):
        if self.i + 1 >= len(self.img_list):
            self.opened = False
            return False, None

        # img = cv2.flip(cv2.imread(os.path.join(self.dir, self.img_list[self.i])),0)
        img = cv2.imread(os.path.join(self.dir, self.img_list[self.i]))
        self.i += 1
        if img is None:
            self.opened = False
            return False, None
        return True, img

    def isOpened(self):
        return self.opened

    def release(self):
        ...

    def get(self, code):
        img = cv2.imread(os.path.join(self.dir, self.img_list[0]))
        if code == cv2.CAP_PROP_FRAME_HEIGHT:
            return img.shape[0]
        if code == cv2.CAP_PROP_FRAME_WIDTH:
            return img.shape[1]

    def set(self, code, val):
        if code == cv2.CAP_PROP_POS_FRAMES:
            self.i = val
        return