"""Simple AutoMated vectorization of "sentences" of "words" to create custom
embeddings based on n-gram co-occurence.
"""

import sys

import numpy as np
from scipy.linalg import norm


NGRAM_SIZE = 3


def tokenize(sentence):
    # TODO: Use stemmer
    return sentence.strip().split()


def get_ngrams(sentence):
    ngrams = []
    for i in xrange(len(sentence)):
        ngram = sentence[i:i + NGRAM_SIZE]
        if len(ngram) < NGRAM_SIZE:
            break
        ngrams.append(tuple(ngram))
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
    return vector / norm(vector)


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
        sentences = []
        for sentence in fp:
            words = tokenize(sentence)
            sentences.append(words)
    vectors = vectorize(sentences)
    for sentence, vector in zip(sentences, vectors):
        # print sentence, "\n", vector, "\n"
        print sentence, "\n"
        for sentence2, vector2 in zip(sentences, vectors):
            score = vector.T.dot(vector2)
            print "\t", score, sentence2
    np.savez(vector_filename, vectors=vectors, sentences=sentences)


if __name__ == '__main__':
    main()
