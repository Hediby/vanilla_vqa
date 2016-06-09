from os import listdir
from os.path import join
from time import time
import json
from collections import Counter
from time import time

import tensorflow as tf
import numpy as np
from utils import load_dataset, load_vocab, get_batch

class ImageQA(object):
    def __init__(self, dh, dq, da, max_q, Nq, Na, infer=False):
        self.dh = dh
        self.dq = dq
        self.da = da
        self.max_q = max_q
        self.Nq = Nq
        self.Na = Na
        self.infer = infer
        
        with tf.device('/cpu:0'):
            self.qemb_W = tf.get_variable('qemb_w',
                                          initializer=tf.random_uniform([self.Nq, self.dq], -0.1, 0.1))
        self.aemb_W = tf.get_variable(name='aemb_w',
                                      initializer=tf.random_uniform([self.dh, self.Na], -0.1, 0.1))
        self.aemb_b = tf.get_variable(name='aemb_b',
                                      initializer=tf.zeros([self.Na]))

        self.gru = tf.nn.rnn_cell.GRUCell(self.dh)
    def build_model(self,batch_size):
        
        p_question = tf.placeholder(tf.int32, 
                                    [None, self.max_q],
                                    name="p_question")
        p_answer = tf.placeholder(tf.float32, 
                                  [None,self.Na],
                                  name="p_answer")
        p_question_mask = tf.placeholder(tf.int32,
                                         [self.max_q, None, None],
                                         name="p_question_mask")
        
        state = tf.zeros([batch_size, self.gru.state_size])
        states = []
        outputs = []
        for j in range(self.max_q):
            with tf.device('/cpu:0'):
                question_emb = tf.nn.embedding_lookup(self.qemb_W, p_question[:,j])
            if j>0:
                tf.get_variable_scope().reuse_variables()
            output,state = self.gru(question_emb, state)
            states.append(state)
            outputs.append(output)

        output = tf.pack(outputs) # (max_words_q, batch_size, 4*dim_hidden)
        output_final = tf.reduce_sum(tf.mul(output, tf.to_float(p_question_mask)),0) # (batch_size, 2*dim_hidden)

        answer_logits = tf.nn.xw_plus_b(output_final,
                                        self.aemb_W,
                                        self.aemb_b) # (batch_size, num_answer)
        answer_pred = tf.argmax(answer_logits,1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(answer_logits, p_answer) # (batch_size, )
        loss = tf.reduce_mean(cross_entropy)
        train_op = tf.train.AdamOptimizer().minimize(loss)
        loss_summary = tf.scalar_summary("loss",loss,name="loss")
        output = {'train_op':train_op,
                 'loss':loss,
                 'question':p_question,
                 'mask':p_question_mask,
                 'answer':p_answer,
                 'answer_pred':answer_pred,
                 'loss_summary':loss_summary}
        return output
    
def test(step,verbose=None):
    N_test = len(q_test)
    n_batches = N_test // batch_size
    acc = []
    for idx in range(n_batches):
        if verbose:
            if idx%20==0:
                print("%d/%d - accuracy = %1.3f"%(idx,n_batches, np.mean(acc)))
        begin = idx*batch_size
        end = min((idx+1)*batch_size, N_test)
        Q, mask, A = get_batch(begin,end,q_test,a_test,batch_size,max_q,Na)
        a_pred = sess.run(model_outputs['answer_pred'], 
                          feed_dict={model_outputs['question']:Q,
                                     model_outputs['mask']:mask, 
                                     model_outputs['answer']:A})
        equals = 1*np.equal(A.argmax(axis=1),a_pred)
        equals = list(equals[:end-begin])
        acc += equals
    acc = tf.reduce_mean(tf.to_float(acc))
    acc_s = tf.scalar_summary("acc_tf",acc,name="acc_tf")
    acc,acc_s = sess.run([acc,acc_s])
    writer.add_summary(acc_s,step)
    return acc

if __name__=="__main__":
    q_train = load_dataset('datasets/coco/train/questions.idxs')
    q_test = load_dataset('datasets/coco/test/questions.idxs')
    a_train = load_dataset('datasets/coco/train/answers.idxs')
    a_test = load_dataset('datasets/coco/test/answers.idxs')

    q_i2w, q_w2i = load_vocab('datasets/coco/train/questions.vocab')
    a_i2w, a_w2i = load_vocab('datasets/coco/train/answers.vocab')

    max_q = len(max(q_train, key=lambda x:len(x)))+1
    Nq = len(q_i2w)
    Na = len(a_i2w)

    dh = 50 #LSTM hidden state dimension
    dq = 75 #Question embedding dimension
    da = 50 #Answer embedding dimension
    batch_size = 64

    print("Graph initialization")
    tf.reset_default_graph()
    model = ImageQA(dh,dq,da,max_q,Nq,Na,batch_size)
    model_outputs = model.build_model(batch_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    saver = tf.train.Saver(max_to_keep=100)
    writer = tf.train.SummaryWriter('/home/hbenyounes/vqa/model1/tf_log', sess.graph)

    init = tf.initialize_all_variables()
    sess.run(init)

    #Train
    q_train = np.array(q_train)
    a_train = np.array(a_train)
    with tf.device('/gpu:0'):
        n_epochs = 50
        N_train = len(q_train)
        n_batches = N_train // batch_size + 1
        for epoch in range(n_epochs):
            epoch_loss = []
            times = 0.
            indexes = np.arange(N_train)
            np.random.shuffle(indexes)
            q_train = q_train[indexes]
            a_train = a_train[indexes]
            for idx in range(n_batches):
                tic = time()
                if idx%(n_batches//10)==0:
                    print("Epoch %d - %d/%d : loss = %1.4f - time = %1.3fs"%(epoch,idx,
                                                                             n_batches,np.mean(epoch_loss),
                                                                             times/((n_train//10)*batch_size)))
                    times = 0.
                begin = idx*batch_size
                end = min((idx+1)*batch_size, N_train)
                Q, mask, A = get_batch(begin,end,q_train,a_train,batch_size,max_q,Na)
                _,l,l_s = sess.run([model_outputs['train_op'],
                                    model_outputs['loss'],
                                    model_outputs['loss_summary']], 
                                   feed_dict={model_outputs['question']:Q,
                                              model_outputs['mask']:mask,
                                              model_outputs['answer']:A})
                epoch_loss.append(l)
                writer.add_summary(l_s,idx+epoch*n_batches)
                times += time() - tic
            with tf.device('/cpu:0'):
                test_acc = test((1+epoch)*n_batches)
                print("Epoch %d - Test accuracy = %1.3f" % (epoch+1, test_acc))
            saver.save(sess, join('/home/hbenyounes/vqa/saved_models/','model'), global_step=epoch)