import tensorflow as tf
tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
from PIL import Image
#%matplotlib widget ## doesn't work with multiple virtual environments
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import torchvision.transforms.functional as TF
import random
import os
import glob

class PathsDictMaker:
    def __init__(self, folder_maps, base_folder=""):
        if base_folder:
            assert(os.path.isdir(base_folder))
            self._make_folder_dicts(folder_maps, base_folder) 
    
    def _make_folder_dicts(self, folder_maps, base_folder):
        self._folder_dicts = {}
        for mode,dct in folder_maps.items():
            tmp = {}
            for key, val in dct.items():
                tmp[key] = os.path.join(base_folder, val)
                try:
                    assert(os.path.isdir(tmp[key]))
                except:
                    print("check the path given for %s in folder_maps"%key)
            self._folder_dicts[mode] = tmp
    
    def make_paths_dicts(self, file_extension=".tif"):
        paths_dicts = {}
        for mode,dct in self._folder_dicts.items():
            tmp = {}
            for key,val in dct.items():
                tmp[key] = sorted(glob.glob(val + "/*" + file_extension))
            paths_dicts[mode] = tmp
        for mode,dct in paths_dicts.items():
            assert(len(dct["images"]) == len(dct["labels"]))
            print("Number of examples in %s dataset: "%mode, len(dct["images"]))
        return paths_dicts

def _extract_image(datapoint):
    img_pil = tf.py_function(_read_images_pil, [datapoint["images"]], tf.uint8)
    mask_pil = tf.py_function(_read_images_pil, [datapoint["labels"]], tf.uint8)
    return img_pil, mask_pil

def _read_images_pil(filename, size):
    size=(556,556)
    image = Image.open(filename).resize(size)
    image = np.asarray(image)
    return image

class ImageInputPipeline:
    def __init__(self, folder_maps, file_extension=".tif", base_folder=""):
        paths_dict_maker = PathsDictMaker(folder_maps, base_folder)
        paths_dicts = paths_dict_maker.make_paths_dicts(file_extension)
        self._train_paths_ds = tf.data.Dataset.from_tensor_slices(paths_dicts["train"])
        self._val_paths_ds = tf.data.Dataset.from_tensor_slices(paths_dicts["val"])

    def _extract_image_(self, image, label):
        print("exe")
        augment=True
        size=(556,556)
        img_pil = Image.open(image.decode('utf-8')).resize(size)
        mask_pil = Image.open(label.decode('utf-8')).resize(size)
        if augment:
            return self._augment_image(img_pil, mask_pil)
        else:
            return self._pil_to_np(img_pil, mask_pil)
    
    def _augment_image(self, image_pil, mask_pil):
        ## start with affine transform since it works on pil images
        angle = random.randint(-30, 30)
        translate =(random.randint(-50, 50), random.randint(-50, 50))
        scale = 0.9 + random.random()*0.2
        shear = random.randint(-6, 6)
        img_trf = TF.affine(image_pil, angle, translate, scale, shear, resample=0, fillcolor=None)
        mask_trf = TF.affine(mask_pil, angle, translate, scale, shear, resample=0, fillcolor=None)
        ## convert images to np
        img_trf, mask_trf = self._pil_to_np(img_trf, mask_trf)
        ## perform elastic transforms - adds channel dimension to mask_trf
        alpha = random.randint(0,700)
        sigma = 5.75 + random.random()*1.25
        img_trf, mask_trf = elastic_transform_git_mod(img_trf, mask_trf[..., None], alpha, sigma)
        ## brightness adjustments - converts img_trf to tensor
        img_trf = tf.image.random_brightness(img_trf, max_delta=0.15)
        ## contrast adjustments
        img_trf = tf.image.random_contrast(img_trf, lower = 0.15, upper = 1.5)
        return img_trf, mask_trf
        
    def _pil_to_np(self, img_pil, mask_pil):
        img  = np.array(img_pil).astype("float32")/255.
        mask = np.array(mask_pil)
        return np.expand_dims(img, axis=-1), mask
    
    def _preprocess(self, datapoint, augment, size):
        img, mask = _extract_image(datapoint) #tf.py_function(self._extract_image, [datapoint], [tf.float32, tf.int8])
        return img, mask
        
    def train_input_fn(self, size, batch_size):
        self._train_paths_ds.map(lambda datapoint: self._preprocess(datapoint, augment=True, size=size), num_parallel_calls=AUTOTUNE)
        return self._train_paths_ds.batch(batch_size).repeat().shuffle(4000).prefetch(AUTOTUNE)

if __name__ == "__main__":
    base_dir = "/mnt/sda/deep_learning/CSE527_FinalProject-master/images"
    folder_maps ={
        "train": {
            "images": "train",
            "labels": "label",
        },
        "val": {
            "images": "test",
            "labels": "test_label"
        }
    }
    input_pipeline = ImageInputPipeline(folder_maps, ".tif", base_dir)
    for image, label in input_pipeline.train_input_fn(size=(556,556), batch_size=4).take(1):
        print("Image shape: ", image)
        print("Label: ", label)