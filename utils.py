import numpy as np

def load_vocab(vocab_path):
    i2w = []
    w2i = {}
    for i,l in enumerate(open(vocab_path,'r',encoding='latin1')):
        l = l.strip()
        i2w.append(l)
        w2i[l] = i
    return i2w, w2i


def load_dataset(idxs_path):
    dataset = []
    for l in open(idxs_path, 'r',encoding='latin1'):
        dataset.append([int(i) for i in l.strip().split()])
    return dataset

def get_batch(begin,end,X,Y,
              batch_size,max_q,Na):
    Q = np.zeros((batch_size, max_q), 'int32')
    mask = np.zeros((max_q,batch_size), 'int32')
    for i,s in enumerate(X[begin:end]):
        Q[i] = np.pad(s, (0,max_q-len(s)), 'constant')
        mask[len(s)-1,i] = 1
    ans = np.zeros((batch_size,Na),'int32')
    for i,a in enumerate(Y[begin:end]):
        ans[i,a] = 1
    mask = mask[:,:,None]
    return Q,mask,ans

def read_features(path,n_max):
    feats = {}
    for n,l in enumerate(open(path,encoding='latin1')):
        l = l.strip().split(';')
        idx = l[0]
        feat = [float(x) for x in l[1].split()]
        feats[idx] = feat
        if n>n_max:
            break
    return feats

def load_emb_matrix(q_i2w, embeddings,vector_size):
    out = []
    c = set()
    for w in q_i2w:
        if w in embeddings:
            out.append(embeddings[w])
        else:
            c.add(w)
            out.append(np.zeros((vector_size,)))
    return (np.array(out),c)

class Dataset(object):
    def __init__(self, f_path, i_path, q_path, a_path, n_max=np.Inf):
        self.f_path = f_path
        self.i_path = i_path
        self.q_path = q_path
        self.a_path = a_path
        print('Parse features file')
        self.images_lines = {}
        self.images_features = []
        for i,l in enumerate(open(self.f_path)):
            l = l.split(';')
            self.images_lines[l[0]] = i
            self.images_features.append([float(x) for x in l[1].split()])
            if i>= n_max:
                break
        print('Parse questions file')
        
        q_data = load_dataset(self.q_path)
        self.max_q = len(max(q_data, key=lambda x:len(x)))
        print('Parse answers file')
        a_data = load_dataset(self.a_path)
        self.data = []
        for q_id,q,a in zip(open(i_path),q_data,a_data):
            q_id = q_id.strip()
            try:
                l_num = self.images_lines[q_id]
            except:
                continue
            datum = (l_num,q,a)
            self.data.append(datum)
        self.data = np.array(self.data, dtype=object)
        del q_data,a_data
        self.N = len(self.data)
        self.indexes = np.arange(self.N)
    
    def __iter__(self):
        return self
    
    def batch_gen(self,batch_size=64):
        np.random.shuffle(self.indexes)
        n_batches = self.N // batch_size
        for batch_id in range(n_batches):
            begin = batch_id*batch_size
            end = min((batch_id+1)*batch_size, self.N)
            B = self.data[self.indexes[begin:end]]
            B = [(self.images_features[b[0]],b[1],b[2]) for b in B]
            yield B