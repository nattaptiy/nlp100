from lib import build_morphs,Morph,Chunk,clean_sym

fname = 'neko.txt.cabocha'

for chunks in build_morphs():
    for chunk in chunks:
        if chunk.dst != -1:
            src = clean_sym(chunk)
            dst = clean_sym(chunks[chunk.dst])
            if src != '' and dst != '':
                print('{}\t{}'.format(src, dst))