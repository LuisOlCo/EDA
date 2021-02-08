import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
# Functions used in step 1
def split_page_in_sentences(page):
    '''
    Split page text into sentences, deletes empty items in list
    '''
    return [sentence.strip() for sentence in page.split('.') if sentence!='']

def parse_chapter_page(line):
    '''
    Parse the chapter and page in the chapter from the first sentence of the page
    '''
    pattern = '[0-9]'
    result = re.match(pattern,line[0])

    if result:
        chapter = line[0]
        if re.match(pattern,line[3]):
            page = line[2:4]
        else:
            page = line[2]
        return int(chapter),int(page)

    elif line[:2] == 'G-':
        chapter = line[0]
        if re.match(pattern,line[3]):
            page = line[2:4]
        else:
            page = line[2]
        return chapter,int(page)

    elif line[:2] == 'I-':
        chapter = line[0]
        if re.match(pattern,line[3]):
            page = line[2:4]
        else:
            page = line[2]
        return chapter,int(page)

    else:
        return 'Intro',None


# Functions used in step 3A
def get_bow_corpus(token_to_idx,chapter_counter):
    '''
    Transforms the chapter_counter dictionary into a list of tuple lists,
    this is the format required by Gensim to compute TD-IDF or LDA

    @Input: 1.- Dictionary tokens with their respectives index
            2.- Dictionary with the frequency of each word in each chapter
    '''
    bow_corpus = []
    for chapter in chapter_counter:
        chapter_list = []
        for word in chapter_counter[chapter]:
            chapter_list.append((token_to_idx[word],chapter_counter[chapter][word]))
        bow_corpus.append(chapter_list)
    return bow_corpus

def relevant_words_TDIDF_per_chapter(top_n_relevant_words,corpus_tfidf,idx_to_token):
    '''
    Returns back the top n most TD-IDF relevant words in a chapter
    @Input: 1.- number of most relevant words
            2.- TD-IDF values for each chapters
            3.- idx_to_token dictionary to identify the word
    '''
    top_n_relevant_words = 10
    values = []
    most_relevant_words = [None]*top_n_relevant_words

    for word in corpus_tfidf:
        values.append(word[1])

    values = sorted(values)
    top_values = values[-top_n_relevant_words:]

    for word in corpus_tfidf:
        if word[1] in top_values:
            index = top_values.index(word[1])
            most_relevant_words[index] = idx_to_token[word[0]]

    return most_relevant_words

def bi_grams(corpus):
    '''
    Retuns a list of tuples with the most common bi-grams in the corpus
    @Input: 1.- Corpus is a list of all sentences in the corpus (stopwords not included)
    '''
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    # vectorizer.vocabulary_.items() --> returns a list of tuples with the brigram and its freq
    X = vectorizer.fit_transform(corpus)
    #X.toarray()
    # sum the repetitions of each bi-gram
    freq_bigrams = X.sum(axis=0)
    bigrams = [(word, freq_bigrams[0, freq]) for word, freq in vectorizer.vocabulary_.items()]
    bigrams =sorted(bigrams, key = lambda x: x[1], reverse=True)
    return bigrams
