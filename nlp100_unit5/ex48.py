from lib import build_morphs,Morph,Chunk,clean_sym,chk_pos,morphs_pos,find_kaku,find_sa
result = 'result.txt'

with open(result, mode='w',encoding='utf-8') as out_file:
    for chunks in build_morphs():
        for chunk in chunks:

            if (len(morphs_pos(chunk,'名詞')) > 0):

                # print(clean_sym(chunk),end='')

                # 根へのパスを出力
                dst = chunk.dst
                while dst != -1:
                    # print(' -> ' + clean_sym(chunks[dst]),end='')
                    out_file.write(' -> ' + clean_sym(chunks[dst]))
                    dst = chunks[dst].dst
                # print('')
                out_file.write('\n')