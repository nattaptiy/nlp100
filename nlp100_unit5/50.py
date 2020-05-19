fname = 'neko.txt.cabocha'

class Morph:
    def __init__(self, surface, base, pos, pos1):
        '''初期化'''
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def __str__(self):
        return 'surface[{}]\tbase[{}]\tpos[{}]\tpos1[{}]'\
            .format(self.surface, self.base, self.pos, self.pos1)


def build_morphs():
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

for i, morphs in enumerate(build_morphs(), 1):
    if i == 3:
        for morph in morphs:
            print(morph)
        break
