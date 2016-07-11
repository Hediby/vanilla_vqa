import tensorflow as tf

class ImageQA(object):
    def __init__(self, dh, dq, da, di, max_q, Nq, Na, cell='rnn',trainable_embeddings=True):
        self.dh = dh
        self.dq = dq
        self.da = da
        self.di = di
        self.max_q = max_q
        self.Nq = Nq
        self.Na = Na
        self.cell = cell
        
        with tf.device('/cpu:0'):
            self.qemb_W = tf.get_variable('qemb_w',
                                          initializer=tf.random_uniform([self.Nq, self.dq], -0.1, 0.1),
                                          trainable = trainable_embeddings)
        self.aemb_W = tf.get_variable(name='aemb_w',
                                      initializer=tf.random_uniform([self.dh, self.Na], -0.1, 0.1))
        self.aemb_b = tf.get_variable(name='aemb_b',
                                      initializer=tf.zeros([self.Na]))
        self.Wi = tf.get_variable(name='Wi', shape=[self.di, self.dq],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.bi = tf.get_variable(name='bi',
                                      initializer=tf.zeros([self.dq]))
        
        if self.cell == 'rnn':
            self.recur = tf.nn.rnn_cell.RNNCell(self.dh)
        elif self.cell == 'lstm':
            self.recur = tf.nn.rnn_cell.LSTMCell(self.dh)
        elif self.cell == 'gru':
            self.recur = tf.nn.rnn_cell.GRUCell(self.dh)
        else:
            raise NotImplementedError
        
    def build_model(self,batch_size):
        
        p_image = tf.placeholder(tf.float32,
                                [None, self.di],
                                 name="p_image")
        
        p_keep_prob = tf.placeholder(tf.float32, name='p_keep_prob')
        
        p_question = tf.placeholder(tf.int32, 
                                    [None, self.max_q],
                                    name="p_question")
        p_answer = tf.placeholder(tf.float32, 
                                  [None,self.Na],
                                  name="p_answer")
        p_question_mask = tf.placeholder(tf.int32,
                                         [self.max_q+1, None, None],
                                         name="p_question_mask")
        
        image_proj = tf.nn.xw_plus_b(p_image,self.Wi,self.bi,name='image_proj')
        image_proj_drp = tf.nn.dropout(image_proj, p_keep_prob)
        
        state = self.recur.zero_state(batch_size, tf.float32)
        states = []
        outputs = []
        for j in range(self.max_q+1):
            if j==0:
                output,state = self.recur(image_proj_drp, state)
            else:
                with tf.device('/cpu:0'):
                    question_emb = tf.nn.embedding_lookup(self.qemb_W, p_question[:,j-1])
                    question_emb_drp = tf.nn.dropout(question_emb, p_keep_prob)
                tf.get_variable_scope().reuse_variables()
                output,state = self.recur(question_emb_drp, state)
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
                  'keep_prob':p_keep_prob,
                 'answer_pred':answer_pred,
                 'loss_summary':loss_summary,
                 'image':p_image}
        return output