import numpy as np
from IPython import embed
from sets import Set
from tqdm import tqdm
GLOVE_PATH="/home/ec2-user/ScrambleTests/dataset/GloVe/glove.840B.300d.txt"

PTB_PATH="/tmp/ptb-copy/vocab.txt"
OUT_PATH="/tmp/ptb-v4/"

L = 10
T = 80
N = 42000
# 2L + T


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

import numpy as np
def generate_line():
    seq = np.random.choice(a07, L, replace=True)
    input_seq = seq.tolist()
    input_seq.extend([a8] * (T-1))
    input_seq.append(a9)
    input_seq.extend([a8] * L)

    output_seq = [a8] * (T + L)
    output_seq.extend(seq.tolist())

    # sanity check
    assert len(input_seq) == len(output_seq)

    return " ".join(input_seq) + " " + " ".join(output_seq)

for output_file in ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]:
    # NB(demi): probability of overlap is little
    f = open(output_file)
    for i in tqdm(range(N), desc=output_file):
        line = generate_line()
        f.write(" " + line + " \n")
    f.close()

