from gevent import monkey
monkey.patch_all()
from collections import Counter

import json
import bottle
import numpy as np
from word2vec import Word2Vec
import MovieQA
import random
# from bottle import bottle.route, run, template, static_file, request,

mqa = MovieQA.DataLoader()
story_raw, qa = mqa.get_story_qa_data('train', 'plot')
imdb_keys = story_raw.keys()

w2v = Word2Vec(extension='plt_mar14', postprocess=True)

def clean_plot(imdb_key):
    text_raw = story_raw[imdb_key]
    text = ' '.join(sentence for sentence in text_raw)
    text = text.split()
    return text

@bottle.route('/static/<filename:path>', 'GET')
def serve_pictures(filename):
    return bottle.static_file(filename, root='./static/')

@bottle.route('/tfidf/<imdb_key>/<query_word>')
def tfidf_plot(imdb_key, query_word):
    # TODO: Implement
    pass

@bottle.route('/w2v/<imdb_key>/<query_word>')
def w2v_plot(imdb_key, query_word):
    if not imdb_key in imdb_keys:
        return bottle.template("key_not_found", imdb_key=imdb_key)

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
def hello():
    bottle.redirect('/w2v/tt0241527/Harry')


bottle.run(host='0.0.0.0', port=8088, debug=True)
