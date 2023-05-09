import re
from wordcloud import STOPWORDS 
import os
import csv

pattern = r"[!♪#$%&()*+,-./:;<=>?@[\]^_`{|}~]"

def clean_line(line):
    global pattern
    line = line.strip()
    line = re.sub(pattern, '', line)
    line = line.lower()
    return line

def counting_freq(content):
    content = content.split()
    word_dict = {}
    for word in content:
        if word not in STOPWORDS:
            word_dict[word] = word_dict.get(word, 0) + 1
    word_dict = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse = True))
    return word_dict

def remove_empty(PATH):
    genre_list = os.listdir(PATH)
    for gen in genre_list:
        genre_PATH = os.path.join(PATH, gen)
        try:
            list_lyric = os.listdir(genre_PATH)
        except:
            continue
        empty_song = 0
        for lyric in list_lyric:
            lyric_PATH = os.path.join(genre_PATH, lyric)
            with open(lyric_PATH, 'r', encoding='utf-8') as lf:
                for count, line in enumerate(lf):
                    pass
            if count == 0:
                empty_song += 1
                os.remove(lyric_PATH)
    print("========= total removed songs: {}".format(empty_song))

# find empty files
def check_emptyFiles(PATH):
    list_genre = os.listdir(PATH)
    for gen in list_genre:
        genre_PATH = os.path.join(PATH, gen)
        list_lyric = os.listdir(genre_PATH)
        empty_files = []
        for lyric in list_lyric:
            lyric_PATH = os.path.join(genre_PATH, lyric)
            with open(lyric_PATH, 'r', encoding='utf-8') as lf:
                for count, line in enumerate(lf):
                    pass
            if count == 0:
                empty_files.append(lyric)

        print('     {}: {} \nTotal: {}'.format(gen, empty_files, len(empty_files)))


def check_duplicate(PATH):
    list_genre = os.listdir(PATH)
    for gen in list_genre:
        genre_PATH = os.path.join(PATH, gen)
        list_lyric = os.listdir(genre_PATH)
        songlist, dup = [], []
        count = 0
        for lyric in list_lyric:
            lyric_PATH = os.path.join(genre_PATH, lyric)
            with open(lyric_PATH, 'r', encoding='utf-8') as f:
                fl = f.readlines()[1:]
                if fl not in songlist:
                    songlist.append(fl)
                else:
                    dup.append(lyric)
                    count += 1
        print(gen, ': ', count)
        print(dup)
