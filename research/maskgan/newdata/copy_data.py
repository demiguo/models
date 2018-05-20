import numpy as np
from IPython import embed
from sets import Set
GLOVE_PATH="/home/ec2-user/ScrambleTests/dataset/GloVe/glove.840B.300d.txt"

PTB_PATH="/tmp/ptb-copy/vocab.txt"
OUT_PATH="/tmp/ptb-v4/"

T = 80
min_len = 3
max_len=9
N = 42000


def get_words(n=100):
    words = Set([])
    f = open(PTB_PATH)
    for line in f:
        line_words = line,replace("\n", "").split(" ")
        for word in line_words:
            words.add(word)
    f.close()
    words = list(words)
    import random
    random.shuffle(words)
    assert n+2 <= len(words)
    return words[:n], words[n], words[n+1]
    

a07, a8, a9 = get_words(100)

#output = open("copy_data_ptb_T%d_min%d_v3.txt" % (T,min_len), "w") 
output = open("/tmp/ptb-copy-v4/ptb.valid.txt", "w")
num_words = len(words)
for i in range(N):
    line = []
    for j in range(10):
        wid = np.random.randint(0, high=num_words-1)
        line.append(words[wid])
    for j in range(T-1):
        line.append(a8)
    line.append(a9)
    for j in range(10):
        line.append(line[j])
    line_str = " ".join(line)
    output.write(" " + line_str + " \n")    
output.close()

