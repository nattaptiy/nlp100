#20
import gzip
import json
fname = 'jawiki-country.json.gz'

with gzip.open(fname,mode='rt',encoding='utf8') as data_file:
    for line in data_file:
        data_json = json.loads(line)
        if data_json['title'] == 'イギリス':
            print(data_json['text'])
            break

#21
import re

def extract_UK():
    with gzip.open(fname,mode='rt',encoding='utf8') as data_file:
        for line in data_file:
            data_json = json.loads(line)
            if data_json['title'] == 'イギリス':
                return data_json['text']

    raise ValueError('イギリスの記事が見つからない')

pattern = re.compile(r'''
    ^   # 行頭
    (   # キャプチャ対象のグループ開始
    .*  # 任意の文字0文字以上
    \[\[Category:
    .*  # 任意の文字0文字以上
    \]\]
    .*  # 任意の文字0文字以上
    )   # グループ終了
    $   # 行末
    ''', re.MULTILINE + re.VERBOSE)

result = pattern.findall(extract_UK())

for line in result:
    print(line)

#22
pattern = re.compile(r'''
    ^       # 行頭
    .*      # 任意の文字0文字以上
    \[\[Category:
    (       # キャプチャ対象のグループ開始
    .*?     # 任意の文字0文字以上、非貪欲マッチ（貪欲にすると後半の'|'で始まる装飾を巻き込んでしまう）
    )       # グループ終了
    (?:     # キャプチャ対象外のグループ開始
    \|.*    # '|'に続く0文字以上
    )?      # グループ終了、0か1回の出現
    \]\]
    .*      # 任意の文字0文字以上
    $       # 行末
    ''', re.MULTILINE + re.VERBOSE)

# 抽出
result = pattern.findall(extract_UK())

# 結果表示
for line in result:
    print(line)

#23
pattern = re.compile(r'''
    ^       # 行頭
    (={2,}) # キャプチャ対象、2個以上の'='
    \s*     # 余分な0個以上の空白（'哲学'や'婚姻'の前後に余分な空白があるので除去）
    (.+?)   # キャプチャ対象、任意の文字が1文字以上、非貪欲（以降の条件の巻き込み防止）
    \s*     # 余分な0個以上の空白
    \1      # 後方参照、1番目のキャプチャ対象と同じ内容
    .*      # 任意の文字が0文字以上
    $       # 行末
    ''', re.MULTILINE + re.VERBOSE)

# 抽出
result = pattern.findall(extract_UK())

# 結果表示
for line in result:
    level = len(line[0]) - 1    # '='の数-1
    print('{indent}{sect}({level})'.format(
        indent='\t' * (level - 1), sect=line[1], level=level))

#24
pattern = re.compile(r'''
    (?:File|ファイル)   # 非キャプチャ、'File'か'ファイル'
    :
    (.+?)   # キャプチャ対象、任意の文字1文字以上、非貪欲
    \|
    ''', re.VERBOSE)

# 抽出
result = pattern.findall(extract_UK())

# 結果表示
for line in result:
    print(line)

#25
pattern = re.compile(r'''
    ^\{\{基礎情報.*?$   # '{{基礎情報'で始まる行
    (.*?)       # キャプチャ対象、任意の0文字以上、非貪欲
    ^\}\}$      # '}}'の行
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

# 基礎情報テンプレートの抽出
contents = pattern.findall(extract_UK())

# 抽出結果からのフィールド名と値の抽出条件コンパイル
pattern = re.compile(r'''
    ^\|         # '|'で始まる行
    (.+?)       # キャプチャ対象（フィールド名）、任意の1文字以上、非貪欲
    \s*         # 空白文字0文字以上
    =
    \s*         # 空白文字0文字以上
    (.+?)       # キャプチャ対象（値）、任意の1文字以上、非貪欲
    (?:         # キャプチャ対象外のグループ開始
        (?=\n\|)    # 改行+'|'の手前（肯定の先読み）
        | (?=\n$)   # または、改行+終端の手前（肯定の先読み）
    )           # グループ終了
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

# フィールド名と値の抽出
fields = pattern.findall(contents[0])

# 辞書にセット
result = {}
keys_test = []      # 確認用の出現順フィールド名リスト
for field in fields:
    result[field[0]] = field[1]
    keys_test.append(field[0])

# 確認のため表示（確認しやすいようにkeys_testを使ってフィールド名の出現順にソート）
for item in sorted(result.items(),
        key=lambda field: keys_test.index(field[0])):
    print(item)

#26

def remove_markup(target):

    # 除去対象の正規表現のコンパイル
    pattern = re.compile(r'''
        \'{2,5} # 2〜5個の'
        ''', re.MULTILINE + re.VERBOSE)

    # 空文字に置換
    return pattern.sub('', target)


# 基礎情報テンプレートの抽出条件のコンパイル
pattern = re.compile(r'''
    ^\{\{基礎情報.*?$   # '{{基礎情報'で始まる行
    (.*?)       # キャプチャ対象、任意の0文字以上、非貪欲
    ^\}\}$      # '}}'の行
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

# 基礎情報テンプレートの抽出
contents = pattern.findall(extract_UK())

# 抽出結果からのフィールド名と値の抽出条件コンパイル
pattern = re.compile(r'''
    ^\|         # '|'で始まる行
    (.+?)       # キャプチャ対象（フィールド名）、任意の1文字以上、非貪欲
    \s*         # 空白文字0文字以上
    =
    \s*         # 空白文字0文字以上
    (.+?)       # キャプチャ対象（値）、任意の1文字以上、非貪欲
    (?:         # キャプチャ対象外のグループ開始
        (?=\n\|)    # 改行+'|'の手前（肯定の先読み）
        | (?=\n$)   # または、改行+終端の手前（肯定の先読み）
    )           # グループ終了
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

# フィールド名と値の抽出
fields = pattern.findall(contents[0])

# 辞書にセット
result = {}
keys_test = []      # 確認用の出現順フィールド名リスト
for field in fields:
    result[field[0]] = remove_markup(field[1])
    keys_test.append(field[0])

# 確認のため表示（確認しやすいようにkeys_testを使ってフィールド名の出現順にソート）
for item in sorted(result.items(),
        key=lambda field: keys_test.index(field[0])):
    print(item)

#27
def remove_markup2(target):
    '''マークアップの除去
    強調マークアップと内部リンクを除去する

    引数：
    target -- 対象の文字列
    戻り値：
    マークアップを除去した文字列
    '''

    # 強調マークアップの除去
    pattern = re.compile(r'''
        (\'{2,5})   # 2〜5個の'（マークアップの開始）
        (.*?)       # 任意の1文字以上（対象の文字列）
        (\1)        # 1番目のキャプチャと同じ（マークアップの終了）
        ''', re.MULTILINE + re.VERBOSE)
    target = pattern.sub(r'\2', target)

    # 内部リンクの除去
    pattern = re.compile(r'''
        \[\[        # '[['（マークアップの開始）
        (?:         # キャプチャ対象外のグループ開始
            [^|]*?  # '|'以外の文字が0文字以上、非貪欲
            \|      # '|'
        )??         # グループ終了、このグループが0か1出現、非貪欲
        ([^|]*?)    # キャプチャ対象、'|'以外が0文字以上、非貪欲（表示対象の文字列）
        \]\]        # ']]'（マークアップの終了）
        ''', re.MULTILINE + re.VERBOSE)
    target = pattern.sub(r'\1', target)

    return target


# 基礎情報テンプレートの抽出条件のコンパイル
pattern = re.compile(r'''
    ^\{\{基礎情報.*?$   # '{{基礎情報'で始まる行
    (.*?)       # キャプチャ対象、任意の0文字以上、非貪欲
    ^\}\}$      # '}}'の行
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

# 基礎情報テンプレートの抽出
contents = pattern.findall(extract_UK())

# 抽出結果からのフィールド名と値の抽出条件コンパイル
pattern = re.compile(r'''
    ^\|         # '|'で始まる行
    (.+?)       # キャプチャ対象（フィールド名）、任意の1文字以上、非貪欲
    \s*         # 空白文字0文字以上
    =
    \s*         # 空白文字0文字以上
    (.+?)       # キャプチャ対象（値）、任意の1文字以上、非貪欲
    (?:         # キャプチャ対象外のグループ開始
        (?=\n\|)    # 改行+'|'の手前（肯定の先読み）
        | (?=\n$)   # または、改行+終端の手前（肯定の先読み）
    )           # グループ終了
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

# フィールド名と値の抽出
fields = pattern.findall(contents[0])

# 辞書にセット
result = {}
keys_test = []      # 確認用の出現順フィールド名リスト
for field in fields:
    result[field[0]] = remove_markup2(field[1])
    keys_test.append(field[0])

# 確認のため表示（確認しやすいようにkeys_testを使ってフィールド名の出現順にソート）
for item in sorted(result.items(),
        key=lambda field: keys_test.index(field[0])):
    print(item)

#28
def remove_markup3(target):
    '''マークアップの除去
    MediaWikiマークアップを可能な限り除去する

    引数：
    target -- 対象の文字列
    戻り値：
    マークアップを除去した文字列
    '''

    # 強調マークアップの除去
    pattern = re.compile(r'''
        (\'{2,5})   # 2〜5個の'（マークアップの開始）
        (.*?)       # 任意の1文字以上（対象の文字列）
        (\1)        # 1番目のキャプチャと同じ（マークアップの終了）
        ''', re.MULTILINE + re.VERBOSE)
    target = pattern.sub(r'\2', target)

    # 内部リンク、ファイルの除去
    pattern = re.compile(r'''
        \[\[        # '[['（マークアップの開始）
        (?:         # キャプチャ対象外のグループ開始
            [^|]*?  # '|'以外の文字が0文字以上、非貪欲
            \|      # '|'
        )*?         # グループ終了、このグループが0以上出現、非貪欲
        ([^|]*?)    # キャプチャ対象、'|'以外が0文字以上、非貪欲（表示対象の文字列）
        \]\]        # ']]'（マークアップの終了）
        ''', re.MULTILINE + re.VERBOSE)
    target = pattern.sub(r'\1', target)

    # Template:Langの除去        {{lang|言語タグ|文字列}}
    pattern = re.compile(r'''
        \{\{lang    # '{{lang'（マークアップの開始）
        (?:         # キャプチャ対象外のグループ開始
            [^|]*?  # '|'以外の文字が0文字以上、非貪欲
            \|      # '|'
        )*?         # グループ終了、このグループが0以上出現、非貪欲
        ([^|]*?)    # キャプチャ対象、'|'以外が0文字以上、非貪欲（表示対象の文字列）
        \}\}        # '}}'（マークアップの終了）
        ''', re.MULTILINE + re.VERBOSE)
    target = pattern.sub(r'\1', target)

    # 外部リンクの除去  [http://xxxx] 、[http://xxx xxx]
    pattern = re.compile(r'''
        \[http:\/\/ # '[http://'（マークアップの開始）
        (?:         # キャプチャ対象外のグループ開始
            [^\s]*? # 空白以外の文字が0文字以上、非貪欲
            \s      # 空白
        )?          # グループ終了、このグループが0か1出現
        ([^]]*?)    # キャプチャ対象、']'以外が0文字以上、非貪欲（表示対象の文字列）
        \]          # ']'（マークアップの終了）
        ''', re.MULTILINE + re.VERBOSE)
    target = pattern.sub(r'\1', target)

    # <br>、<ref>の除去
    pattern = re.compile(r'''
        <           # '<'（マークアップの開始）
        \/?         # '/'が0か1出現（終了タグの場合は/がある）
        [br|ref]    # 'br'か'ref'
        [^>]*?      # '>'以外が0文字以上、非貪欲
        >           # '>'（マークアップの終了）
        ''', re.MULTILINE + re.VERBOSE)
    target = pattern.sub('', target)

    return target


# 基礎情報テンプレートの抽出条件のコンパイル
pattern = re.compile(r'''
    ^\{\{基礎情報.*?$   # '{{基礎情報'で始まる行
    (.*?)       # キャプチャ対象、任意の0文字以上、非貪欲
    ^\}\}$      # '}}'の行
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

# 基礎情報テンプレートの抽出
contents = pattern.findall(extract_UK())

# 抽出結果からのフィールド名と値の抽出条件コンパイル
pattern = re.compile(r'''
    ^\|         # '|'で始まる行
    (.+?)       # キャプチャ対象（フィールド名）、任意の1文字以上、非貪欲
    \s*         # 空白文字0文字以上
    =
    \s*         # 空白文字0文字以上
    (.+?)       # キャプチャ対象（値）、任意の1文字以上、非貪欲
    (?:         # キャプチャ対象外のグループ開始
        (?=\n\|)    # 改行+'|'の手前（肯定の先読み）
        | (?=\n$)   # または、改行+終端の手前（肯定の先読み）
    )           # グループ終了
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

# フィールド名と値の抽出
fields = pattern.findall(contents[0])

# 辞書にセット
result = {}
keys_test = []      # 確認用の出現順フィールド名リスト
for field in fields:
    result[field[0]] = remove_markup3(field[1])
    keys_test.append(field[0])

# 確認のため表示（確認しやすいようにkeys_testを使ってフィールド名の出現順にソート）
for item in sorted(result.items(),
        key=lambda field: keys_test.index(field[0])):
    print(item)

#29

import urllib.parse, urllib.request

# 基礎情報テンプレートの抽出条件のコンパイル
pattern = re.compile(r'''
    ^\{\{基礎情報.*?$   # '{{基礎情報'で始まる行
    (.*?)       # キャプチャ対象、任意の0文字以上、非貪欲
    ^\}\}$      # '}}'の行
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

# 基礎情報テンプレートの抽出
contents = pattern.findall(extract_UK())

# 抽出結果からのフィールド名と値の抽出条件コンパイル
pattern = re.compile(r'''
    ^\|         # '|'で始まる行
    (.+?)       # キャプチャ対象（フィールド名）、任意の1文字以上、非貪欲
    \s*         # 空白文字0文字以上
    =
    \s*         # 空白文字0文字以上
    (.+?)       # キャプチャ対象（値）、任意の1文字以上、非貪欲
    (?:         # キャプチャ対象外のグループ開始
        (?=\n\|)    # 改行+'|'の手前（肯定の先読み）
        | (?=\n$)   # または、改行+終端の手前（肯定の先読み）
    )           # グループ終了
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)

# フィールド名と値の抽出
fields = pattern.findall(contents[0])

# 辞書にセット
result = {}
for field in fields:
    result[field[0]] = remove_markup3(field[1])

# 国旗画像の値を取得
fname_flag = result['国旗画像']

# リクエスト生成
url = 'https://www.mediawiki.org/w/api.php?' \
    + 'action=query' \
    + '&titles=File:' + urllib.parse.quote(fname_flag) \
    + '&format=json' \
    + '&prop=imageinfo' \
    + '&iiprop=url'

# MediaWikiのサービスへリクエスト送信
request = urllib.request.Request(url,
    headers={'User-Agent': 'NLP100_Python(@segavvy)'})
connection = urllib.request.urlopen(request)

# jsonとして受信
data = json.loads(connection.read().decode())

# URL取り出し
url = data['query']['pages'].popitem()[1]['imageinfo'][0]['url']
print(url)