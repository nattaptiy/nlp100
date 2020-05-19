from lib import build_morphs,Morph,Chunk,clean_sym,chk_pos,morphs_pos,find_kaku,find_sa

for chunks in build_morphs():
    for chunk in chunks:

        if (len(morphs_pos(chunk,'名詞')) > 0):

            print(clean_sym(chunk),end='')

            # 根へのパスを出力
            dst = chunk.dst
            while dst != -1:
                print(' -> ' + clean_sym(chunks[dst]),end='')
                dst = chunks[dst].dst
            print('')