import re


class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def __str__(self):
        return 'surface[{}]\tbase[{}]\tpos[{}]\tpos1[{}]'\
            .format(self.surface, self.base, self.pos, self.pos1)

class Chunk:

    def __init__(self):
        self.morphs = []
        self.srcs = []
        self.dst = -1

    def __str__(self):
        surface = ''
        for morph in self.morphs:
            surface += morph.surface
        return '{}\tsrcs{}\tdst[{}]'.format(surface, self.srcs, self.dst)

def build_morphs(fname = 'neko.txt.cabocha'):
    with open(fname,'r',encoding='utf-8') as inputfile:

        chunks = dict()
        idx = -1

        for line in inputfile:

            if line == 'EOS\n':
                if len(chunks) > 0:

                    sorted_tuple = sorted(chunks.items(), key=lambda x: x[0])
                    yield list(zip(*sorted_tuple))[1]
                    chunks.clear()

                else:
                    yield []

            elif line[0] == '*':

                cols = line.split(' ')
                idx = int(cols[1])
                dst = int(re.search(r'(.*?)D', cols[2]).group(1))

                if idx not in chunks:
                    chunks[idx] = Chunk()
                chunks[idx].dst = dst

                if dst != -1:
                    if dst not in chunks:
                        chunks[dst] = Chunk()
                    chunks[dst].srcs.append(idx)

            else:
                cols = line.split('\t')
                res_cols = cols[1].split(',')

                chunks[idx].morphs.append(
                    Morph(
                        cols[0],
                        res_cols[6],
                        res_cols[0],
                        res_cols[1]
                    )
                )

def clean_sym(target):
    result = ''
    for morph in target.morphs:
        if morph.pos != '記号':
            result += morph.surface
    return result

def chk_pos(target, pos):
    for morph in target.morphs:
        if morph.pos == pos:
            return True
        return False
def morphs_pos(target, pos, pos1=''):
    if len(pos1) > 0:
        return [res for res in target.morphs if (res.pos == pos) and (res.pos1 == pos1)]
    else:
        return [res for res in target.morphs if res.pos == pos]

def find_kaku(target):
    prts_tmp = morphs_pos(target,'助詞')
    if len(prts_tmp) > 1:
        kaku = morphs_pos(target,'助詞', '格助詞')
        if len(kaku) > 0:
            prts_tmp = kaku

    if len(prts_tmp) > 0:
        return prts_tmp[-1].surface
    else:
        return ''

def find_sa(target):
    for i, morph in enumerate(target.morphs[0:-1]):
        if (morph.pos == '名詞') and (morph.pos1 == 'サ変接続') and (target.morphs[i + 1].pos == '助詞') and (target.morphs[i + 1].surface == 'を'):
            return morph.surface + target.morphs[i + 1].surface
    return ''

def noun_change(target, ch, ending=False):
    result = ''
    for morph in target.morphs:
        if morph.pos != '記号':
            if morph.pos == '名詞':
                result += ch
                if ending:
                    return result
                ch = ''
            else:
                result += morph.surface
    return result