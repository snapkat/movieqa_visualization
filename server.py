from gevent import monkey
monkey.patch_all()
from collections import Counter

import json
import bottle
import numpy as np
from word2vec import Word2Vec
from tfidf import TfIdf
import MovieQA
import random
import math
import sys
# from bottle import bottle.route, run, template, static_file, request,

algos = ['w2v', 'tfidf', 's_tfidf']

mqa = MovieQA.DataLoader()
story_raw, qa = mqa.get_story_qa_data('train', 'plot')
imdb_keys = story_raw.keys()

w2v = Word2Vec(extension='plt_mar14', postprocess=True)
tfidf = TfIdf(story_raw)

def clean_plot(imdb_key):
    text_raw = story_raw[imdb_key]
    text = ' '.join(sentence for sentence in text_raw)
    text = text.split()
    return text

@bottle.route('/static/<filename:path>', 'GET')
def serve_pictures(filename):
    return bottle.static_file(filename, root='./static/')


@bottle.route('/<algo>')
@bottle.route('/<algo>/')
def w2v_movies(algo):
    if not algo in algos:
        return bottle.template("error", message= 'Algorithm %s not found.' % algo)

    return display_movies(algo)


@bottle.route('/<algo>/<imdb_key>/')
@bottle.route('/<algo>/<imdb_key>/<query_word>')
def plot(algo, imdb_key, query_word=None):
    if not imdb_key in imdb_keys:
        return bottle.template("error", message= 'Imdb key %s not found.' % imdb_key)
    if not algo in algos:
        return bottle.template("error", message= 'Algorithm %s not found.' % algo)

    if not query_word:
        plot = clean_plot(imdb_key)
        query_word = plot[0]

    if algo == 'tfidf':
        return tfidf_plot(imdb_key, query_word)

    return w2v_plot(imdb_key, query_word)


def score_tfidf(q):
    global tfidf
    # Sentence vectors for question, answers, plot
    question_vec = tfidf.getSentenceVector(q.imdb_key, q.question)
    answer_matrix = tfidf.getSentenceVectors(q.imdb_key, q.answers)
    plot_matrix = tfidf.getSentenceVectors(
        q.imdb_key, story_raw[q.imdb_key])

    score = plot_matrix
    '''
    qscore = plot_matrix.dot(question_vec).reshape(-1, 1)
    ascore = plot_matrix.dot(answer_matrix.T)
    score = ascore + qscore
    '''
    return score

def predict_tfidf(q):
    global tfidf
    score = score_tfidf(q)
    prediction = np.unravel_index(score.argmax(), score.shape)
    return prediction[1], prediction[0], score[prediction]

def tfidf_plot(imdb_key, query_word):
    global w2v # TEMP USELESS FOR QUERY WORD
    global tfidf
    if not imdb_key in imdb_keys:
        return bottle.template("key_not_found", imdb_key=imdb_key)

    origPlot, cleanPlot = tfidf.getCleanPlot(imdb_key)
    weights = tfidf.getWordVectors(imdb_key, cleanPlot)
    weights = np.rint(weights/np.max(weights)*255) # numWord * 1
    weights = np.reshape(weights, (-1, 1))
    params = {'imdb_key':imdb_key, 'plot':origPlot, 'clean_words':cleanPlot , 'weights':weights, 
              'query_word':None, 'rand': random.random(), 'w2v': False}
    return bottle.template("plot_display", **params)


def w2v_plot(imdb_key, query_word):
    if not imdb_key in imdb_keys:
        return bottle.template("key_not_found", imdb_key=imdb_key)

    global w2v
    plot = clean_plot(imdb_key)

    clean_words = w2v.clean_words(plot)
    query_embed = w2v.get_raw_word_embeddings([query_word])
    plot_embed = w2v.get_raw_word_embeddings(plot)

    weights = np.dot(plot_embed, query_embed.T)
    weights = np.rint(weights/np.max(weights)*255)

    params = {'imdb_key':imdb_key, 'plot':plot, 'clean_words':clean_words, 'weights':weights, 
              'query_word':w2v.tokenized_as(query_word), 'rand': random.random(), 'w2v': True}
    return bottle.template("plot_display", **params)


@bottle.route('/')
def default():
    bottle.redirect('/tfidf/')

def display_movies(algo):
    num_posters = 20
    rows = int(math.ceil(num_posters/6.0))
    return bottle.template("movie_select", imdb_keys=imdb_keys[:num_posters], rows=rows, rand= random.random(), w2v=(algo=='w2v') )

bottle.run(host='0.0.0.0', port=8088, debug=True)
