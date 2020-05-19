from lib import build_morphs,Morph,Chunk

fname = 'neko.txt.cabocha'

for i, chunks in enumerate(build_morphs(), 1):

    # 8文目を表示
    if i == 8:
        for j, chunk in enumerate(chunks):
            print('[{}]{}'.format(j, chunk))
        break
