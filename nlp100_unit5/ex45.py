from lib import build_morphs,Morph,Chunk,clean_sym,chk_pos,morphs_pos,find_kaku
result = 'result.txt'

with open(result, mode='w',encoding='utf-8') as out_file:
    for chunks in build_morphs():
        for chunk in chunks:
            verbs = morphs_pos(chunk,'動詞')
            if len(verbs) < 1:
                continue
            prts = []
            for src in chunk.srcs:
                if len(find_kaku(chunks[src])) > 0:
                    prts.append(chunks[src])
            if len(prts) < 1:
                continue

            prts.sort(key=lambda x: find_kaku(x))

            # print('{}\t{}'.format(verbs[0].base,' '.join(find_kaku(prt) for prt in prts)))

            out_file.write('{}\t{}\n'.format(verbs[0].base, ' '.join(find_kaku(prt) for prt in prts)))
