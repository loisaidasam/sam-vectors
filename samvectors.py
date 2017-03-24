
import sys

import numpy as np


NGRAM_SIZE = 3


def get_ngrams(sentence):
    words = sentence.split()
    # TODO: Use stemmer
    ngrams = []
    for i in xrange(len(words)):
        ngram = words[i:i + NGRAM_SIZE]
        if len(ngram) < NGRAM_SIZE:
            break
        ngrams.append(ngram)
    return ngrams


def get_all_ngrams(sentences):
    sentence_ngrams = []
    all_ngrams = set()
    for sentence in sentences:
        ngrams = get_ngrams(sentence)
        sentence_ngrams.append(ngrams)
        all_ngrams |= set(ngrams)
    return list(all_ngrams), sentence_ngrams


def normalize_vector(vector):
    # TODO: Implement
    return vector


def vectorize(sentences):
    all_ngrams, sentence_ngrams = get_all_ngrams(sentences)
    vectors = []
    for ngrams in sentence_ngrams:
        vector = np.zeros(len(all_ngrams))
        for ngram in ngrams:
            vector[all_ngrams.index(ngram)] = 1
        vector = normalize_vector(vector)
        vectors.append(vector)
    return vectors


def main():
    corpus_filename = sys.argv[1]
    vector_filename = sys.argv[2]
    with open(corpus_filename, 'r') as fp:
        sentences = [row for row in fp]
    vectors = vectorize(sentences)
    np.savez(vector_filename, vectors=vectors, sentences=sentences)


if __name__ == '__main__':
    main()
