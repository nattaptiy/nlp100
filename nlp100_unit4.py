#30
print('#30')

filename_mecab = 'neko.txt.mecab'

def create_words():
    with open(filename_mecab,encoding='utf-8') as target:
        words=[]
        for line in target:
            col_by_tab = line.split('\t')
            if(len(col_by_tab)<2):
                continue
            col_by_space = col_by_tab[1].split(',')

            words.append({
                'surface' : col_by_tab[0],
                'base' : col_by_space[6],
                'pos' : col_by_space[0],
                'pos1' : col_by_space[1]
            })

            if col_by_space[1]=='句点':
                yield words
                words =[]

lines = create_words()
for line in lines:
   print(line)

#31
print('#31')

lines = create_words()
verbs = set()
for line in lines:
    for word in line:
        if word['pos']=='動詞':
            verbs.add(word['surface'])

print(verbs)

#32
print('#32')

lines = create_words()
verbs = set()
for line in lines:
    for word in line:
        if word['pos']=='動詞':
            verbs.add(word['base'])

print(verbs)

#33
print('#33')

lines = create_words()
a_no_b = set()
for line in lines:
    if len(line)>=3:
        for i in range(1,len(line)-1):
            if line[i]['surface']=='の' and line[i-1]['pos']=='名詞' and line[i+1]['pos']=='名詞':
                a_no_b.add(line[i-1]['surface']+line[i]['surface']+line[i+1]['surface'])

print(a_no_b)

#34
print('#34')

lines = create_words()
long_nouns = set()
for line in lines:
    nouns = []
    for word in line:
        if word['pos'] == '名詞':
            nouns.append(word['surface'])
        else:
            if len(nouns) >= 2:
                long_nouns.add(''.join(nouns))
            nouns = []
    if len(nouns) >= 2:
        long_nouns.add(''.join(nouns))

print(long_nouns)

#35
print('#35')

lines = create_words()
word_counter = dict()
for line in lines:
   for word in line:
       if word['surface'] in word_counter:
           word_counter[word['surface']]+=1
       else :
           word_counter[word['surface']]=1

sort_word_counter = sorted(word_counter.items(),key=lambda x:-x[1])
print(sort_word_counter)

#36
print('#36')

top_ten_word = sort_word_counter[0:10]
print(top_ten_word)
words=[]
counts=[]
for i in range(0,10):
    words.append(top_ten_word[i][0])
    counts.append(top_ten_word[i][1])

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.bar(
    range(0,10),
    counts,
    align='center'
)

plt.xticks(
    range(0,10),
    words,
    fontproperties=FontProperties(fname='C:\Windows\Fonts\msgothic.ttc')
)

plt.show()

plt.close()

#37
print('#37')

lines = create_words()
neko_word_counter = dict()
for line in lines:
    if '猫' in [word['surface'] for word in line]:
       for word in line:
           if word['surface']!='猫':
               if word['surface'] in neko_word_counter:
                   neko_word_counter[word['surface']]+=1
               else :
                   neko_word_counter[word['surface']]=1

sort_neko_word_counter = sorted(neko_word_counter.items(),key=lambda x:-x[1])
print(sort_neko_word_counter)

top_ten_neko_word = sort_neko_word_counter[0:10]
print(top_ten_neko_word)
words=[]
counts=[]
for i in range(0,10):
    words.append(top_ten_neko_word[i][0])
    counts.append(top_ten_neko_word[i][1])

plt.bar(
    range(0,10),
    counts,
    align='center'
)

plt.xticks(
    range(0,10),
    words,
    fontproperties=FontProperties(fname='C:\Windows\Fonts\msgothic.ttc')
)

plt.show()

plt.close()

#38
print('#38')

counts= list(zip(*sort_word_counter))[1]

plt.hist(
    counts,
    bins=30,
    range=(1,30)
)

plt.xlim(xmin=1, xmax=30)

plt.show()

plt.close()

#39

plt.scatter(
    range(1,len(counts)+1),
    counts
)

plt.xlim(1,len(counts)+1)
plt.ylim(1,counts[0])

plt.xscale('log')
plt.yscale('log')

plt.show()

plt.close()