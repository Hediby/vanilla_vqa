from os import listdir, mkdir
from os.path import join, exists
from time import time
import json


import tensorflow as tf
import numpy as np


import skimage
from skimage.transform import resize
from scipy.misc import imread
from collections import Counter
from time import time

from utils import load_dataset, load_vocab, Dataset
from vislstm import ImageQA


def load_emb_matrix(q_i2w, embeddings):
    out = []
    c = set()
    for w in q_i2w:
        if w in embeddings:
            out.append(embeddings[w])
        else:
            c.add(w)
            out.append(np.zeros((vector_size,)))
    return (np.array(out),c)

def create_feed_dict(batch,max_q,Na,batch_size):
    V = np.zeros((batch_size, len(batch[0][0])), 'float32')
    Q = np.zeros((batch_size, max_q), 'int32')
    mask = np.zeros((max_q+1,batch_size), 'int32')
    ans = np.zeros((batch_size,Na),'int32')
    
    for i,(im,s,a) in enumerate(batch):
        V[i] = im
        Q[i] = np.pad(s, (0,max_q-len(s)), 'constant')
        mask[len(s),i] = 1
        ans[i,a] = 1
    mask = mask[:,:,None]
    return V,Q,mask,ans

def test(step,verbose=None):
    acc = []
    test_batches = test_set.batch_gen(batch_size)
    for idx,batch in enumerate(test_batches):    
        if verbose:
            if idx%20==0:
                print("%d - accuracy = %1.3f"%(idx, np.mean(acc)))
        V,Q,mask,ans = create_feed_dict(batch,max_q,Na,batch_size)
        a_pred = sess.run(model_outputs['answer_pred'], 
                          feed_dict={model_outputs['question']:Q,
                                     model_outputs['mask']:mask, 
                                     model_outputs['answer']:ans,
                                     model_outputs['image']:V, 
                                     model_outputs['keep_prob']:keep_prob})
        equals = 1*np.equal(ans.argmax(axis=1),a_pred)
        equals = list(equals[:len(batch)])
        acc += equals
    acc = tf.reduce_mean(tf.to_float(acc))
    acc_s = tf.scalar_summary("acc_tf",acc,name="acc_tf")
    acc,acc_s = sess.run([acc,acc_s])
    writer.add_summary(acc_s,step)
    return acc


if __name__=="__main__":

    model_name = "word2vec_fixed"
    root_path = "/home/hbenyounes/vqa/"
    embedding_path = '/home/hbenyounes/vqa/GoogleNews.model'
    vector_size = 300
    hyperparams = {"dh":2000, 
                   "dq":vector_size,
                   "da":200, 
                   "di":4096,
                   "batch_size":16,
                   "keep_prob":0.5,
                   "cell":"gru"}


    q_i2w, q_w2i = load_vocab('datasets/coco/train/questions.vocab')

    print("Load word2Vec")
    embeddings = {}
    for n,l in enumerate(open(embedding_path,encoding='utf-8')):
        l = l.strip().split()
        w = l[0]
        vec = [float(x) for x in l[1:]]
        embeddings[w] = vec

    emb,c = load_emb_matrix(q_i2w, embeddings)
    del embeddings
    train_set = Dataset("/home/hbenyounes/vqa/datasets/coco/train/images.feat",
                        "/home/hbenyounes/vqa/datasets/coco/train/img_ids.txt",
                        "/home/hbenyounes/vqa/datasets/coco/train/questions.idxs",
                        "/home/hbenyounes/vqa/datasets/coco/train/answers.idxs")


    test_set = Dataset("/home/hbenyounes/vqa/datasets/coco/test/images.feat",
                        "/home/hbenyounes/vqa/datasets/coco/test/img_ids.txt",
                        "/home/hbenyounes/vqa/datasets/coco/test/questions.idxs",
                        "/home/hbenyounes/vqa/datasets/coco/test/answers.idxs")

    if not exists(join(root_path, model_name)):
        mkdir(join(root_path, model_name))

    q_i2w, q_w2i = load_vocab('datasets/coco/train/questions.vocab')
    a_i2w, a_w2i = load_vocab('datasets/coco/train/answers.vocab')
    Nq = len(q_i2w)
    Na = len(a_i2w)

    max_q = train_set.max_q
    Nq = len(q_i2w)
    Na = len(a_i2w)


    dh = hyperparams["dh"] #GRU hidden state dimension
    dq = hyperparams["dq"] #Question embedding dimension
    da = hyperparams["da"] #Answer embedding dimension
    di = hyperparams["di"] #Image dimension
    batch_size = hyperparams["batch_size"]
    keep_prob = hyperparams["keep_prob"]
    cell = hyperparams["cell"]

    print("Graph initialization")
    tf.reset_default_graph()
    model = ImageQA(dh,dq,da,di,max_q,Nq,Na,cell,False)
    model_outputs = model.build_model(batch_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    writer = tf.train.SummaryWriter(join(root_path,model_name,'tf_log'), sess.graph)

    saver = tf.train.Saver(max_to_keep=100)


    # saver.restore(sess, '/home/hbenyounes/vqa/model2/model-8')

    init = tf.initialize_all_variables()
    sess.run(init)


    sess.run(model.qemb_W.assign(emb))

    n_parameters = sum( [np.prod(v.get_shape(),dtype='int') for v in tf.trainable_variables()])
    hyperparams['n_parameters'] = n_parameters

    #Train
    break_all = False
    with tf.device('/gpu:0'):
        n_epochs = 50
        max_test_acc = -np.Inf
        for epoch in range(n_epochs):
            epoch_loss = []
            times = 0.
            n_batches = train_set.N // batch_size
            train_batches = train_set.batch_gen(batch_size)
            for idx,batch in enumerate(train_batches):
                tic = time()
                if idx%(n_batches//10)==0:
                    print("Epoch %d - %d/%d : loss = %1.4f - time = %1.3fs"%(epoch,idx,
                                                                             n_batches,np.mean(epoch_loss),
                                                                             times))
                V,Q,mask,ans = create_feed_dict(batch,max_q,Na,batch_size)
                _,l,l_s = sess.run([model_outputs['train_op'],
                                    model_outputs['loss'],
                                    model_outputs['loss_summary']], 
                                   feed_dict={model_outputs['question']:Q,
                                              model_outputs['mask']:mask,
                                              model_outputs['answer']:ans,
                                              model_outputs['image']:V,
                                              model_outputs['keep_prob']:keep_prob})
                if np.isnan(l):
                    break_all = True
                epoch_loss.append(l)
                writer.add_summary(l_s,idx+epoch*n_batches)
                times += time() - tic
                if break_all:
                    print("Loss is nan at iteration %d" % (idx+n_batches*epoch))
                    break
            if break_all:
                break
            with tf.device('/cpu:0'):
                test_acc = test((1+epoch)*n_batches)
                print("Epoch %d - Test accuracy = %1.3f" % (epoch+1, test_acc))
            if test_acc < max_test_acc:
                saver.save(sess, join(root_path,model_name,'model'), global_step=epoch)
            max_test_acc = max(test_acc, max_test_acc)

    with open(join(root_path, model_name, 'hyperparams'),'w') as f:
        for h in hyperparams:
            f.write("%s = %s\n" % (h, str(hyperparams[h])))
        f.write('\n\nMaximal test accuracy = %1.4f' % max_test_acc)