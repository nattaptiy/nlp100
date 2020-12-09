#00
text='stressed123'[::-1]
print(text)

#01
text='パタトクカシーー'[0:8:2]
print(text)

#02
target1 = 'パトカー'
target2 = 'タクシー'
result = ''
for (a, b) in zip(target1, target2):
    result += a + b
print(result)

#03
text='Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
num_char=[]
for word in text.split():
    count=0
    for letter in word:
        if(letter.isalpha()):
            count+=1
    num_char.append(count)
print(num_char)

#04
text='Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
num_first_only = (1, 5, 6, 7, 8, 9, 15, 16, 19)
words=text.split()
result={}
for (num, word) in enumerate(words, 1):
    if num in num_first_only:
        result[word[0:1]] = num
    else:
        result[word[0:2]] = num
print(result)

#05
def n_gram(target,n):
    return [ target[idx:idx + n] for idx in range(len(target) - n + 1)]

text='I am an NLPer'
print(n_gram(text.split(),2))

#06

set_x = set(n_gram('paraparaparadise', 2))
print('X:' + str(set_x))
set_y = set(n_gram('paragraph', 2))
print('Y:' + str(set_y))

# 和集合
set_or = set_x | set_y
print('和集合:' + str(set_or))

# 積集合
set_and = set_x & set_y
print('積集合:' + str(set_and))

# 差集合
set_sub = set_x - set_y
print('差集合:' + str(set_sub))

# 'se'が含まれるか？
print('seがXに含まれる:' + str('se' in set_x))
print('seがYに含まれる:' + str('se' in set_y))

#07

def format_str(x, y, z):
    return '{hour}時の{target}は{value}'.format(hour=x, target=y, value=z)

x = 12
y = '気温'
z = 22.4
print(format_str(x, y, z))

#08
def cipher(target):
    result = ''
    for c in target:
        if c.islower():
            result += chr(219 - ord(c))
        else:
            result += c
    return result


text = 'This is a test. Mary had a little lamb. I live in USA.'

result = cipher(text)
print('暗号化:' + result)

result2 = cipher(result)
print('復号化:' + result2)

#09

import random


def Typoglycemia(target):
    result = []
    for word in target.split(' '):
        if len(word) <= 4:
            result.append(word)
        else:
            chr_list = list(word[1:-1])
            random.shuffle(chr_list)
            result.append(word[0] + ''.join(chr_list) + word[-1])

    return ' '.join(result)

text = 'I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'
result = Typoglycemia(text)
print('変換結果:' + result)