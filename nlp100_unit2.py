#10
from datetime import date

fname = 'popular-names.txt'
with open(fname) as lines:
    count = sum(1 for line in lines)
print(count)

#11
with open(fname) as data_file:
    for line in data_file:
        print(line.replace('\t', ' '), end='')

#12
with open(fname) as data_file, \
        open('col1.txt', mode='w') as col1_file, \
        open('col2.txt', mode='w') as col2_file:
    for line in data_file:
        cols = line.split('\t')
        col1_file.write(cols[0] + '\n')
        col2_file.write(cols[1] + '\n')

#13
with open('col1.txt') as col1_file, \
        open('col2.txt') as col2_file, \
        open('merge.txt', mode='w') as out_file:

    for col1_line, col2_line in zip(col1_file, col2_file):
        out_file.write(col1_line.rstrip() + '\t' + col2_line.rstrip() + '\n')

#14
fname = 'merge.txt'
n = 5

with open(fname) as data_file:
    for i, line in enumerate(data_file):
        if i >= n:
            break
        print(line.rstrip())

#15
fname = 'merge.txt'
n = 5

if n > 0:
    with open(fname) as data_file:
        lines = data_file.readlines()

    for line in lines[-n:]:
        print(line.rstrip())

#16
import math

fname = 'merge.txt'
n = 3

with open(fname) as data_file:
    lines = data_file.readlines()

count = len(lines)
unit = math.ceil(count / n)  # 1ファイル当たりの行数

for i, offset in enumerate(range(0, count, unit), 1):
    with open('child_{:02d}.txt'.format(i), mode='w') as out_file:
        for line in lines[offset:offset + unit]:
            out_file.write(line)

#17
fname = 'popular-names.txt'
with open(fname) as data_file:

    set_ken = set()
    for line in data_file:
        cols = line.split('\t')
        set_ken.add(cols[0])

for n in set_ken:
    print(n)

#18
fname = 'popular-names.txt'
lines = open(fname).readlines()
lines.sort(key=lambda line: (line.split('\t')[2]), reverse=True)

for line in lines:
    print(line, end='')

#19
from itertools import groupby
fname = 'popular-names.txt'

# 都道府県名の読み込み
lines = open(fname).readlines()
names = [line.split('\t')[0] for line in lines]

# 都道府県で集計し、(都道府県, 出現頻度)のリスト作成
names.sort()    # goupbyはソート済みが前提
result = [(name, len(list(group))) for name, group in groupby(names)]

# 出現頻度でソート
result.sort(key=lambda name: name[1], reverse=True)

# 結果出力
for name in result:
    print('{name}({count})'.format(name=name[0], count=name[1]))