from os.path import exists
from os import mkdir
from os.path import join
from PIL import Image
import json


import tensorflow as tf
import threading
import roi_pooling_op_grad
module = tf.load_op_library('/Programs/tensorflow/roi_pooling.so')
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.misc import imread, imresize

from utils import load_vocab
from time import time
import datetime

def load_images(ps):
    tic = time()
    images = [imread(p,mode='RGB') for p in ps]
    toc = time()
    print("imread = %1.3fs" % (toc-tic))
    treated_images = []
    sizes = []
    for img in images:
        sizes.append(img.shape[:2])
        treated_img = imresize(img,(448,448),'nearest') / 255.0
        treated_images.append(treated_img)
    tic = time()
    print("Resize / scaling = %1.3fs" % (tic-toc))
    return treated_images,sizes


class Dataset(object):
    def __init__(self,h5_path,image_paths,max_q=None,max_mc=None):
        self.h5 = h5py.File(h5_path,mode='r')
        self.image_ids = self.h5['image_ids'].value
        self.questions = self.h5['questions'].value
        self.multiple_choice = self.h5['multiple_choice'].value
        self.answers = self.h5['answers'].value
        self.bounding_boxes = dict((k,v) for (k,v) in zip(self.h5['img_list'].value, 
                                                          self.h5['bounding_boxes'].value))
        self.N = len(self.image_ids)
        if max_q:
            if max_q<self.questions.shape[1]:
                self.questions = self.questions[:,:max_q]
            else:
                self.questions = np.pad(self.questions,
                                        ((0,0),(0,max_q-self.questions.shape[-1])),
                                        'constant',constant_values=a_w2i['</s>'])
        if max_mc:
            if max_mc<self.multiple_choice.shape[-1]:
                self.multiple_choice = self.multiple_choice[:,:,max_mc]
            else:
                self.multiple_choice = np.pad(self.multiple_choice,
                                              ((0,0),(0,0),(0,max_mc-self.multiple_choice.shape[-1])),
                                              'constant',constant_values=a_w2i['</s>'])
        self.max_q = self.questions.shape[1]
        self.indexes = np.arange(self.N)
        self.image_paths = image_paths
        
    def __iter__(self):
        return self
    
    def batch_gen(self,batch_size=64,shuffle=True):
        def load_image(p):
            img = imread(p,mode='RGB')
            size = img.shape[:2]
            img = imresize(img,(448,448),'nearest') / 255.0
            return img,size
        
        if shuffle:
            np.random.shuffle(self.indexes)
        n_batches = self.N // batch_size
        tiled_batch = np.arange(batch_size)[:,None]
        tiled_batch = np.tile(tiled_batch,(1,100))[:,:,None]
        load_time = 0
        for batch_id in range(n_batches):
            begin = batch_id*batch_size
            end = min((batch_id+1)*batch_size, self.N)
            idxs = self.indexes[begin:end]
            image_ids = self.image_ids[idxs]
            images,sizes = [],[]
            for i in image_ids:
                p = self.image_paths[i]
                img,size = load_image(p)
                images.append(img)
                sizes.append(size)
            images = np.stack(images)
            sizes = np.array(sizes)
            questions = self.questions[idxs]
            lengths = np.sum(np.not_equal(questions, 
                                          a_w2i['</s>']), 
                             axis=1)
            question_mask = np.zeros((self.max_q,batch_size))
            for i,q in enumerate(questions):
                question_mask[lengths[i]-1,i] = 1
            answers = self.answers[idxs]
            multiple_choice = self.multiple_choice[idxs]
            #lengths = np.sum(np.not_equal(multiple_choice,a_w2i['</s>']), axis=-1)
            #multiple_choice = multiple_choice[:,:,:lengths.max()]
            bbs = np.array([self.bounding_boxes[k] for k in image_ids])
            bbs = np.concatenate((tiled_batch,bbs),axis=-1)
            bounding_boxes = np.reshape(bbs, (bbs.shape[0]*bbs.shape[1],bbs.shape[2]))
            yield (images,questions,question_mask,answers,multiple_choice,bounding_boxes,sizes)

def test_threading():
    accuracy = 0.
    total_time = 0
    n_batches = val_set.N//batch_size + 1
    t = threading.Thread(target=load_and_enqueue,args=(sess,enqueue_op,False,val_set))
    t.start()
    for idx in range(n_batches):
        tic = time()
        y_pred,answers = sess.run([out_probas,Pl['answers']])
        y_pred = np.argmax(y_pred,axis=1)
        accuracy += np.sum(answers[np.arange(batch_size),y_pred])
        step_time = time()-tic
        total_time += step_time
        eta = total_time*(n_batches-idx)/(idx+1)
        print("\tTest: %d/%d - accuracy = %1.3f -  ETA = %s" % (idx,
                                                                val_set.N/batch_size,
                                                                accuracy/(batch_size*(idx+1)),
                                                                datetime.timedelta(seconds=int(eta))))
    return accuracy / (batch_size*(idx+1))

def load_and_enqueue(sess,enqueue_op,shuffle,dataset):
    batch_gen = dataset.batch_gen(batch_size,shuffle)
    for (images,questions,question_mask,answers,
         multiple_choice,bounding_boxes,sizes) in batch_gen:
        feed_dict = {Ql['images']:images,
                     Ql['boxes']:bounding_boxes,
                     Ql['questions']:questions,
                     Ql['question_mask']:question_mask, 
                     Ql['answers']:answers,
                     Ql['mc']:multiple_choice}
        sess.run(enqueue_op,feed_dict=feed_dict)
if __name__=="__main__":


    image_paths = {}
    root_path = "/srv/data/datasets/mscoco/images/"

    for split in 'train val'.split():
        image_ids_path = "datasets/vqa/"+split+"/img_ids.txt"
        image_ids = set([int(x.strip()) for x in open(image_ids_path).readlines()])
        print(split,len(image_ids))
        for x in image_ids:
            name = 'COCO_'+split+'2014_'+format(x, '012')+'.jpg'
            path = join(root_path,split+"2014",name)
            image_paths[x] = path

    q_i2w, q_w2i = load_vocab('datasets/vqa/train/questions.vocab')
    a_i2w, a_w2i = load_vocab('datasets/vqa/train/answers.vocab')

    train_set = Dataset('datasets/vqa/train/dataset.h5',image_paths)
    max_mc = train_set.multiple_choice.shape[-1]
    max_q = train_set.max_q
    val_set = Dataset('datasets/vqa/val/dataset.h5',image_paths,max_q=max_q,max_mc=max_mc)
    Nq = len(q_i2w)
    Na = len(a_i2w)
    
    tf.reset_default_graph()
    # Read the model
    with open("tensorflow-vgg16/vgg16.tfmodel",
              mode='rb') as f:
        fileContent = f.read()
    graph_def = tf.GraphDef()
    # Put it into my graph_def
    graph_def.ParseFromString(fileContent)
    graph = tf.get_default_graph()

    weights_names = ["import/fc6/weight:0", 
                     "import/fc7/weight:0",
                     "import/fc8/weight:0"]
    biases_names = ["import/fc6/bias:0", 
                    "import/fc7/bias:0",
                    "import/fc8/bias:0"]
    fc_shapes = [4096,4096,1000]
    layer_number = 2
    #di = graph.get_tensor_by_name(weights_names[layer_number-1]).get_shape()[-1].value
    def pool5_tofcX(input_tensor, layer_number=layer_number):
        flatten=tf.reshape(input_tensor,(-1,7*7*512))
        tmp=flatten
        for i in range(layer_number):
            tmp=tf.matmul(tmp, graph.get_tensor_by_name(weights_names[i]))
            tmp=tf.nn.bias_add(tmp, graph.get_tensor_by_name(biases_names[i]))
            tmp = tf.nn.relu(tmp)
        return tmp


    batch_size = 32
    di = fc_shapes[layer_number-1]
    dv = 500
    dq = 300
    dh = 300
    datt = 300
    Nq = train_set.N

    Ql = {}
    Ql['images'] = tf.placeholder(tf.float32, 
                                  [batch_size, 448, 448, 3],
                                  name="images") #batch x width x height x channels
    Ql['boxes'] = tf.placeholder(tf.float32, 
                                 [None,5],
                                 name = "boxes")
    Ql['questions'] = tf.placeholder(tf.int32, 
                                     [batch_size, max_q],
                                     name="question")
    Ql['question_mask'] = tf.placeholder(tf.int32,
                                         [max_q, None],
                                         name="question_mask")
    Ql['mc'] = tf.placeholder(tf.int32,
                              [batch_size, 18,None], 
                              name="mc")
    Ql['answers'] = tf.placeholder(tf.float32, 
                                   [batch_size,18], 
                                   name="answers")

    q = tf.FIFOQueue(100, [tf.float32, tf.float32, 
                           tf.int32, tf.int32, 
                           tf.int32, tf.float32], shapes=[[batch_size,448,448,3],
                                                            [batch_size*100,5],
                                                            [batch_size,max_q],
                                                            [max_q,batch_size],
                                                            [batch_size,18,max_mc],
                                                            [batch_size,18]])

    enqueue_op = q.enqueue([Ql['images'], Ql['boxes'], Ql['questions'], 
                            Ql['question_mask'], Ql['mc'], Ql['answers']])
    Pl = {}
    Pl['images'], Pl['boxes'], Pl['questions'], Pl['question_mask'], Pl['mc'], Pl['answers'] = q.dequeue()


    with tf.variable_scope('image'):
        tf.get_variable('W', shape=[di, dv],
                        initializer=tf.contrib.layers.xavier_initializer())
        tf.get_variable(name='b',
                        initializer=tf.zeros([dv]))

    with tf.variable_scope('question'):
        tf.get_variable('W',
                        initializer=tf.random_uniform([Nq, dq], -0.1, 0.1))

    with tf.variable_scope('attention'):
        tf.get_variable('Wimg',shape=[dv,datt],
                        initializer=tf.contrib.layers.xavier_initializer())
        tf.get_variable('Wstate',shape=[dh,datt],
                        initializer=tf.contrib.layers.xavier_initializer())

    with tf.variable_scope('multiple_choice'):
        tf.get_variable('W',
                        initializer=tf.random_uniform([Na, dh], -0.1, 0.1))

    with tf.variable_scope('multimodal'):
        tf.get_variable('Wv', 
                        shape = [dv,dh], 
                        initializer=tf.contrib.layers.xavier_initializer())
        tf.get_variable(name='bv',
                        initializer=tf.zeros([dh]))
        tf.get_variable('Wq', 
                        shape = [dh,dh], 
                        initializer=tf.contrib.layers.xavier_initializer())
        tf.get_variable(name='bq',
                        initializer=tf.zeros([dh]))
    recurrent = tf.nn.rnn_cell.GRUCell(dh)

    def compute_attention(V,q):
        with tf.variable_scope('attention',reuse=True):
            Wimg = tf.get_variable('Wimg')
            Wstate = tf.get_variable('Wstate')
            Vatt = tf.transpose(tf.tanh(tf.reshape(tf.matmul(tf.reshape(V, 
                                                           (batch_size*100,dv)),
                                                 Wimg),
                                       (100,batch_size,datt))),(1,0,2))
            Hatt = tf.expand_dims(tf.matmul(state,Wstate),1)
            att = tf.batch_matmul(Vatt,Hatt,adj_y=True)
            patt = tf.nn.softmax(att[:,:,0])
            Vpond = tf.mul(V,tf.expand_dims(patt,-1))
            Vt = tf.reduce_sum(Vpond,reduction_indices=1)
            return Vt

    def merge_modalities(Vatt,q_out):
        with tf.variable_scope('multimodal',reuse=True):
            Wv = tf.get_variable('Wv')
            Wq = tf.get_variable('Wq')
            bv = tf.get_variable('bv')
            bq = tf.get_variable('bq')

            xv = tf.nn.relu(tf.nn.xw_plus_b(Vatt,Wv,bv))
            xq = tf.nn.relu(tf.nn.xw_plus_b(q_out,Wq,bq))

            x = tf.tanh(xv + xq)
            return x


    tf.import_graph_def(graph_def, 
                        input_map={'images':Pl['images']})

    out_tensor = graph.get_tensor_by_name("import/conv5_3/Relu:0")
    # Don't do your max pooling, but the roi_pooling
    [out_pool,argmax] = module.roi_pool(out_tensor,
                                        Pl['boxes'],
                                        7,7,1.0/1) # out_pool.shape = N_Boxes x 7 x 7 x 512
    boxes_emb = pool5_tofcX(out_pool,layer_number=layer_number)
    with tf.variable_scope('image',reuse=True):
        W = tf.get_variable("W")
        b = tf.get_variable("b")
    V = tf.tanh(tf.matmul(boxes_emb,W) + b)
    V = tf.reshape(V,(batch_size,100,dv))

    state = recurrent.zero_state(batch_size, tf.float32)
    states = []
    q_out = []
    with tf.variable_scope('question',reuse=True):
        W = tf.get_variable('W')
    for j in range(max_q):
        question_emb = tf.nn.embedding_lookup(W, Pl['questions'][:,j])
        if j>0:
            tf.get_variable_scope().reuse_variables()
        output,state = recurrent(question_emb, state)
        states.append(state)
        q_out.append(output)
    q_out = tf.pack(q_out)
    q_out = tf.reduce_sum(tf.mul(q_out, 
                                 tf.to_float(tf.expand_dims(Pl['question_mask'],-1))),0)
    Vatt = compute_attention(V,q_out)

    x = merge_modalities(Vatt,q_out)

    mc_mask = tf.to_float(tf.not_equal(Pl['mc'],a_w2i['</s>']))
    norm_mask = tf.expand_dims(tf.reduce_sum(mc_mask,reduction_indices=2),-1)
    with tf.variable_scope('multiple_choice'):
        W = tf.get_variable('W')
        mc_emb = tf.nn.embedding_lookup(W, Pl['mc'])
        masked_mc_out = tf.mul(tf.expand_dims(mc_mask,-1),mc_emb)
        mc_out = tf.reduce_sum(masked_mc_out,reduction_indices=2)/norm_mask

    out_scores = tf.batch_matmul(mc_out,tf.expand_dims(x,1),adj_y=True)[:,:,0]
    out_probas = tf.nn.softmax(out_scores)

    normalized_ans = Pl['answers'] / tf.expand_dims(tf.reduce_sum(Pl['answers'],reduction_indices=1),-1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(out_scores,normalized_ans)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer()
    #optimizer = tf.train.GradientDescentOptimizer(0.01)
    gvs = optimizer.compute_gradients(cost)
    # with tf.device('/cpu:0'):
    cost_s = tf.scalar_summary('train loss', cost, name='train_loss')
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad,var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    model_name = "model2"
    model_rootpath = "/home/hbenyounes/vqa/results/vqa/"
    model_path = join(model_rootpath,model_name)
    if not exists(model_path):
        mkdir(model_path)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                            gpu_options=gpu_options))

    writer = tf.train.SummaryWriter(join(model_path,'tf_log'), sess.graph)
    saver = tf.train.Saver(max_to_keep=100)

    #saver.restore(sess,'/home/hbenyounes/vqa/results/vqa/model1/model-15a')
    init = tf.initialize_all_variables()
    sess.run(init)
    
    n_batches = train_set.N//batch_size
    n_epochs = 50
    output_file = open(join(model_path,"output.txt"),'w')
    best_test_acc = -1
    break_all = False
    for epoch in range(1,n_epochs+1):
        t = threading.Thread(target=load_and_enqueue,args=(sess,enqueue_op,True,train_set))
        t.start()
        epoch_loss = []
        total_tic = time()
        train_accuracy = 0.
        for idx in range(n_batches):
            step = idx + (epoch-1)*n_batches
            tic = time()
            _,loss_value,loss_s,y_pred,ans = sess.run([train_op,cost,cost_s,out_probas,Pl['answers']])
            writer.add_summary(loss_s,step)
            toc = time()
            step_time = toc - tic
            total_time = toc - total_tic
            eta = total_time*(n_batches-idx)/(idx+1)
            y_pred = y_pred.argmax(axis=1)
            train_accuracy += np.sum(ans[np.arange(batch_size),y_pred])
            print("Epoch %d/%d - batch %d/%d - loss = %1.3f - accuracy = %1.3f - " \
            "elapsed = %1s - ETA = %s" % (epoch,n_epochs,
                                              idx,n_batches,
                                              loss_value,train_accuracy/(batch_size*(idx+1)),
                                              str(datetime.timedelta(seconds=int(total_time))),
                                              str(datetime.timedelta(seconds=int(eta)))))
            epoch_loss.append(loss_value)
            if np.isnan(loss_value):
                print("Loss is nan, i get out")
                break_all = True
            if break_all:
                break
        if break_all:
            break
        train_accuracy = train_accuracy / (batch_size*(idx+1))
        train_loss = np.mean(epoch_loss)
        output_file.write("Epoch %d - \n\ttrain loss = %1.3f - train accuracy = %1.3f\n" % (epoch,train_loss,train_accuracy))
        output_file.flush()
        print("test")
        if not epoch%5:
            test_acc = test_threading()
            if test_acc > best_test_acc:
                print("Saving model...")
                saver.save(sess, join(model_path,'model'), global_step=epoch)
            output_file.write('\ttest accuracy = %1.3f\n' % test_acc)
            output_file.flush()
            best_test_acc = max(best_test_acc,test_acc)
    output_file.close()
