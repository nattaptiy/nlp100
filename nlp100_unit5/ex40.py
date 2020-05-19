from lib import Morph

fname = 'neko.txt.cabocha'

def build_morphs_simple():
    with open(fname,'r',encoding='utf-8') as inputfile:

        morphs = []
        for line in inputfile:
            if line == 'EOS\n':
                yield morphs
                morphs = []
            else:
                if line[0] == '*':
                    continue
                cols = line.split('\t')
                if len(cols)<2:
                    continue
                second_cols = cols[1].split(',')
                morphs.append(Morph(
                    cols[0],
                    second_cols[6],
                    second_cols[0],
                    second_cols[1]
                ))

for i, morphs in enumerate(build_morphs_simple(), 1):
    if i == 3:
        for morph in morphs:
            print(morph)
        break