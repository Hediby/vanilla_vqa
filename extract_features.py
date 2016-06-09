from os import listdir
from os.path import join
from time import time
import json


import tensorflow as tf
import numpy as np


import skimage
from skimage.transform import resize
from scipy.misc import imread

from collections import Counter
from time import time

from utils import load_dataset, load_vocab, get_batch

def load_image(path):
    # load image
    img = imread(path,mode='RGB')
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    #print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    # resize to 224, 224
    resized_img = resize(crop_img, (224, 224))
    return resized_img

def extract_features(ids, path, output_path, extractor, batch_size=64):
    images_names = dict()
    for p in listdir(path):
        image_id = int(p.split('_')[-1].split('.')[0])
        if image_id in ids:
            images_names[image_id] = p
    batch,names = [],[]
    with open(output_path,'w') as output_file:
        for idx,n in enumerate(images_names):
            p = join(path, images_names[n])
            batch.append(load_image(p))
            names.append(n)
            if len(batch)==batch_size:
                batch = np.stack(batch)
                feed_dict = {images: batch}
                with tf.device('/gpu:0'):
                    features = sess.run(extractor, feed_dict=feed_dict)
                for n,f in zip(names,features):
                    output_file.write("%s;%s\n" % (n, " ".join(str(x) for x in f)))
                print("%d/%d" % (idx,len(images_names)))
                batch, names = [],[]
                output_file.flush()
        if len(batch)>0:
            batch = np.stack(batch)
            feed_dict = {images: batch}
            with tf.device('/gpu:0'):
                features = sess.run(extractor, feed_dict=feed_dict)
            for n,f in zip(names,features):
                output_file.write("%s;%s\n" % (n, " ".join(str(x) for x in f)))
            print("%d/%d" % (idx,len(images_names)))
            output_file.flush()
            
                
if __name__=="__main__":
    train_ids = set([int(j) for j in open('datasets/coco/train/img_ids.txt','r')])

    source = '/home/hbenyounes/vqa/datasets/train2014'
    image_ids = set()
    for p in listdir(source):
        image_id = int(p.split('_')[-1].split('.')[0])
        image_ids.add(image_id)

    print("Source: %s" % source)
    print("Train : %d\nTrain absent from source : %d" % (len(train_ids), len(train_ids-image_ids)))


    test_ids = set([int(j) for j in open('datasets/coco/test/img_ids.txt','r')])
    source = '/home/hbenyounes/vqa/datasets/val2014'
    image_ids = set()
    for p in listdir(source):
        image_id = int(p.split('_')[-1].split('.')[0])
        image_ids.add(image_id)

    print("Source: %s" % source)    
    print("Test : %d\nTest absent from source : %d" % (len(test_ids), len(test_ids-image_ids)))
    
    with open("tensorflow-vgg16/vgg16.tfmodel", mode='rb') as f:
        fileContent = f.read()

    with tf.device('/gpu:0'):
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

        images = tf.placeholder("float", [None, 224, 224, 3])

        tf.import_graph_def(graph_def, input_map={ "images": images })

        graph = tf.get_default_graph()
        out_tensor = graph.get_tensor_by_name("import/Relu_1:0")
        
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                            gpu_options=gpu_options))
    init = tf.initialize_all_variables()
    sess.run(init)
    extract_features(test_ids,'/home/hbenyounes/vqa/datasets/val2014/', 
                     "/home/hbenyounes/vqa/datasets/coco/test/images.feat", out_tensor)
    extract_features(train_ids,'/home/hbenyounes/vqa/datasets/train2014', 
                     "/home/hbenyounes/vqa/datasets/coco/train/images.feat", out_tensor)