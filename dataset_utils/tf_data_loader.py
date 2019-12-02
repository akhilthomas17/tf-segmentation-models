import albumentations as A
import os
import glob
from PIL import Image
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import numpy as np


class PathsDictMaker:
    def __init__(self, folder_maps, base_folder=""):
        if base_folder:
            assert(os.path.isdir(base_folder))
            self._make_folder_dicts(base_folder, folder_maps)
        else:
            self._folder_map = folder_maps
    
    def _make_folder_dicts(self, base_folder, rel_map):
        self._folder_map = {}
        for key, rel_path in rel_map.items():
            path = os.path.join(base_folder, rel_path)
            try:
                assert(os.path.isdir(path))
            except:
                print("check the path given for %s in folder_dict"%key)
            self._folder_map[key] = path
    
    def make_paths_dicts(self, file_extension=".tif"):
        paths_dict = {}
        for key,folder in self._folder_map.items():
            paths_dict[key] = sorted(glob.glob(folder + "/*" + file_extension))
        assert(len(paths_dict["images"]) == len(paths_dict["labels"]))
        print("Number of examples in dataset: ", len(paths_dict["images"]))
        return paths_dict


class ImageInputPipeline:
    def __init__(self, folder_maps, file_extension=".tif", base_folder="", shuffle=True):
        paths_dict_maker = PathsDictMaker(folder_maps, base_folder)
        paths_dicts = paths_dict_maker.make_paths_dicts(file_extension)
        paths_list = list(zip(paths_dicts["images"], paths_dicts["labels"]))
        self._paths_ds = tf.data.Dataset.from_tensor_slices(paths_list)
        if shuffle:
            self._paths_ds = self._paths_ds.repeat().shuffle(4000)
        self._aug = None
        self._crop = None
            
    def _augment_img(self, img, mask, height, width):
        if self._aug is None:
            self._aug = A.Compose([A.RandomBrightness(limit=0.1, p=0.4), A.RandomContrast(limit=0.15, p=0.4), 
                                A.GaussianBlur(blur_limit=7, p=0.2), A.MotionBlur(blur_limit=3, p=0.2),
                                A.GaussNoise(var_limit=0.015, p=0.4),
                                A.Flip(p=0.25), A.RandomRotate90(p=0.25),
                                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, border_mode=0, p=0.8),
                                A.ElasticTransform(alpha_affine=0, alpha=40, sigma=6, border_mode=0, approximate=True, p=0.4),
                                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.5, border_mode=0, p=0.25),
                                A.CenterCrop(height=height, width=width)])
        augmented = self._aug(image=img, mask=mask)
        return augmented["image"], augmented["mask"]
        
    def _preprocess_img(self, img, mask, augment, size):
        img_pil = Image.open(img.numpy()).resize((556,556))
        mask_pil = Image.open(mask.numpy()).resize((556,556))
        img, mask = self._pil_to_np(img_pil, mask_pil)
        if augment:
            img, mask = self._augment_img(img, mask, size[0], size[1])
        else:
            if self._crop is None:
                self._crop = A.Compose([A.CenterCrop(height=size[0], width=size[1])])
            cropped = self._crop(image=img, mask=mask)
            img = cropped["image"]
            mask = cropped["mask"]

        return np.stack((img,)*3, axis=-1), mask[...,None]

    def _pil_to_np(self, img_pil, mask_pil):
        img  = np.array(img_pil).astype("float32")/255.
        mask = np.array(mask_pil)
        return img, mask
    
    def _preprocess_ds(self, datapoint, augment, size):
        img = datapoint[0]
        mask = datapoint[1]
        [img, mask] = tf.py_function(self._preprocess_img, [img, mask, augment, size], [tf.float32, tf.uint8])
        #[img, mask] = tf.py_function(self._load_img, [datapoint["images"], datapoint["labels"]], [tf.float32, tf.uint8])
        img.set_shape([size[0], size[1], 3])
        mask.set_shape([size[0], size[1], 1])
        return img, mask
        
    def _input_fn(self, size, batch_size, augment=False):
        ds = self._paths_ds.map(lambda datapoint: self._preprocess_ds(datapoint, augment=augment, size=size), num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)