from collections import Counter
from os.path import join

def build_vocab(path):
    saving_folder = "/".join(path.split('/')[:-1])
    name = path.split('/')[-1].split('.')[0]
    file = open(path,'r',encoding='latin1')
    sentences = []
    for l in file:
        sentences.append(l.strip().split())
    ct = Counter(x for a in sentences for x in a)
    i2w = sorted(ct, key=ct.get, reverse=True)
    i2w = ['<unk>','<s>', '</s>'] + i2w
    w2i = dict((w,i) for i,w in enumerate(i2w))
    vocab_file = open(join(saving_folder, name+'.vocab'), 'w',encoding='latin1')
    for w in i2w:
        vocab_file.write(w+'\n')
    vocab_file.close()
    return 'done'

def integerify(text_path, vocab_path, pad=False):
    saving_folder = "/".join(text_path.split('/')[:-1])
    name = text_path.split('/')[-1].split('.')[0]
    w2i = {}
    for i,l in enumerate(open(vocab_path,'r',encoding='latin1')):
        l = l.strip()
        w2i[l] = i
    indexes_file = open(join(saving_folder, name+'.idxs'), 'w',encoding='latin1')
    for l in open(text_path, 'r',encoding='latin1'):
        l = l.strip().split() 
        if pad:
            l = ['<s>'] + l + ['</s>']
        idxs = []
        for w in l:
            if w in w2i:
                idxs.append(str(w2i[w]))
            else:
                idxs.append(str(w2i['<unk>']))
        indexes_file.write(' '.join(idxs) + '\n')
    return 'done'

if __name__=="__main__":
    build_vocab('/home/hbenyounes/vqa/datasets/coco/train/questions.txt')
    build_vocab('/home/hbenyounes/vqa/datasets/coco/train/answers.txt')
    integerify('/home/hbenyounes/vqa/datasets/coco/train/questions.txt',
                          '/home/hbenyounes/vqa/datasets/coco/train/questions.vocab')
    integerify('/home/hbenyounes/vqa/datasets/coco/train/answers.txt',
                          '/home/hbenyounes/vqa/datasets/coco/train/answers.vocab')
    integerify('/home/hbenyounes/vqa/datasets/coco/test/questions.txt',
                          '/home/hbenyounes/vqa/datasets/coco/train/questions.vocab')
    integerify('/home/hbenyounes/vqa/datasets/coco/test/answers.txt',
                          '/home/hbenyounes/vqa/datasets/coco/train/answers.vocab')
