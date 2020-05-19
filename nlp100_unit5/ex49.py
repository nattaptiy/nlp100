from lib import build_morphs,Morph,Chunk,clean_sym,chk_pos,morphs_pos,find_kaku,find_sa,noun_change

for chunks in build_morphs():
    index_noun = [i for i in range(len(chunks)) if len(morphs_pos(chunks[i],'名詞')) > 0]

    if len(index_noun) < 2:
        continue

    for i, index_x in enumerate(index_noun[:-1]):
        for index_y in index_noun[i + 1:]:

            visit_y = False
            index_cross = -1
            path_x = []

            dst = chunks[index_x].dst
            while dst != -1:
                if dst == index_y:
                    visit_y = True
                    break
                path_x.append(dst)
                dst = chunks[dst].dst

            if not visit_y:
                dst = chunks[index_y].dst
                while dst != -1:
                    if dst in path_x:
                        index_cross = dst
                        break
                    else:
                        dst = chunks[dst].dst

            if index_cross == -1:
                print(noun_change(chunks[index_x],'X'),end='')
                dst = chunks[index_x].dst
                while dst != -1:
                    if dst == index_y:
                        print(' -> ' +noun_change(chunks[dst],'Y', True),end='')
                        break
                    else:
                        print(' -> ' + clean_sym(chunks[dst]),end='')
                    dst = chunks[dst].dst
                print('')

            else:
                print(noun_change(chunks[index_x],'X'),end='')
                dst = chunks[index_x].dst
                while dst != index_cross:
                    print(' -> ' + clean_sym(chunks[dst]),end='')
                    dst = chunks[dst].dst
                print(' | ',end='')

                print(noun_change(chunks[index_y],'Y'),end='')
                dst = chunks[index_y].dst
                while dst != index_cross:
                    print(' -> ' + clean_sym(chunks[dst]), end='')
                    dst = chunks[dst].dst
                print(' | ', end='')

                print(clean_sym(chunks[index_cross]))