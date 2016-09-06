from collections import Counter
from os.path import join

def build_vocab(path,replacers):
    saving_folder = "/".join(path.split('/')[:-1])
    name = path.split('/')[-1].split('.')[0]
    file = open(path,'r',encoding='latin1')
    sentences = []
    for l in file:
        l = l.lower().strip()
        for r1,r2 in replacers:
            l = l.replace(r1,r2)
        sentences.append(l.split())
    file.close()
    ct = Counter(x.strip() for a in sentences for x in a)
    i2w = sorted(ct, key=ct.get, reverse=True)
    i2w = ['<unk>','<s>', '</s>'] + i2w
    w2i = dict((w,i) for i,w in enumerate(i2w))
    vocab_file = open(join(saving_folder, name+'.vocab'), 'w',encoding='latin1')
    for w in i2w:
        vocab_file.write(w+'\n')
    vocab_file.close()
    return 'done'

def integerify(text_path, vocab_path, pad=False, mc=False, replacers=[]):
    saving_folder = "/".join(text_path.split('/')[:-1])
    name = text_path.split('/')[-1].split('.')[0]
    w2i = {}
    for i,l in enumerate(open(vocab_path,
                              'r',encoding='latin1')):
        l = l.strip()
        w2i[l] = i
    indexes_file = open(join(saving_folder, name+'.idxs'), 
                        'w',encoding='latin1')
    for l in open(text_path, 'r',encoding='latin1'):
        l = l.lower().strip().replace('?','').split()
        if pad:
            l = ['<s>'] + l + ['</s>']
        idxs = []
        prev = ''
        for w in l:
            if mc and w=='|':
                if (prev == w) or (prev == ''):
                    print("Empty mc")
                    idxs.append(str(w2i['<unk>']))
                idxs.append('|')
            elif w in w2i:
                idxs.append(str(w2i[w]))
            else:
                idxs.append(str(w2i['<unk>']))
            if mc and i==len(l):
                if len(w)==0:
                    idxs.append(str(w2i['<unk>']))
            prev = w
        indexes_file.write(' '.join(idxs) + '\n')
    return 'done'

if __name__=="__main__":
    build_vocab('/hhome/hbenyounes/vqa/datasets/vqa/train/questions.txt',replacers=[('?',''),
                                                                                    ('"',''),
                                                                                    ("'",' '),
                                                                                    ('.','')])
    build_vocab('/hhome/hbenyounes/vqa/datasets/vqa/train/mcs.txt',replacers=[('|','')])
    
    print("train questions")
    integerify('/hhome/hbenyounes/vqa/datasets/vqa/train/questions.txt',
               '/hhome/hbenyounes/vqa/datasets/vqa/train/questions.vocab')
    print("train answers")
    integerify('/hhome/hbenyounes/vqa/datasets/vqa/train/answers.txt',
               '/hhome/hbenyounes/vqa/datasets/vqa/train/answers.vocab')
    print("train multiple choice")
    integerify('/hhome/hbenyounes/vqa/datasets/vqa/train/mcs.txt',
               '/hhome/hbenyounes/vqa/datasets/vqa/train/answers.vocab',mc=True)
    print('val questions')
    integerify('/hhome/hbenyounes/vqa/datasets/vqa/val/questions.txt',
               '/hhome/hbenyounes/vqa/datasets/vqa/train/questions.vocab')
    print('val answers')
    integerify('/hhome/hbenyounes/vqa/datasets/vqa/val/answers.txt',
               '/hhome/hbenyounes/vqa/datasets/vqa/train/answers.vocab')
    print('val multiple choice')
    integerify('/hhome/hbenyounes/vqa/datasets/vqa/val/mcs.txt',
               '/hhome/hbenyounes/vqa/datasets/vqa/train/answers.vocab',mc=True)