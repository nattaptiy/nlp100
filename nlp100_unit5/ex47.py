from lib import build_morphs,Morph,Chunk,clean_sym,chk_pos,morphs_pos,find_kaku,find_sa
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

            sa_prts = ''
            for prt in prts:
                sa_prts = find_sa(prt)
                if len(sa_prts) > 0:
                    prts_remove = prt
                    break
            if len(sa_prts) < 1:
                continue

            prts.remove(prts_remove)
            prts.sort(
                key=lambda x: find_kaku(x)
            )

            # print('{}\t{}\t{}'.format(sa_prts+verbs[0].base,' '.join(find_kaku(prt) for prt in prts),' '.join(clean_sym(prt) for prt in prts)))

            out_file.write('{}\t{}\t{}'.format(sa_prts + verbs[0].base,' '.join(find_kaku(prt) for prt in prts),' '.join(clean_sym(prt) for prt in prts)))