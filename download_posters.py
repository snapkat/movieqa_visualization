import json
import MovieQA
import random
import math
import sys
import urllib
from urllib2 import Request, urlopen, URLError
# from bottle import bottle.route, run, template, static_file, request,

algos = ['w2v', 'tfidf', 's_tfidf']

mqa = MovieQA.DataLoader()
story_raw, qa = mqa.get_story_qa_data('train', 'plot')
imdb_keys = story_raw.keys()
'''
for key in imdb_keys:
    urllib.urlretrieve("http://www.gunnerkrigg.com//comics/00000001.jpg", "00000001.jpg")
'''

for key in imdb_keys[10:]:
    request = Request('http://www.omdbapi.com/?i=%s'%key)

    try:
        response = urlopen(request)
        json_resp = response.read()
        print json.loads(json_resp)['Poster']
        urllib.urlretrieve(json.loads(json_resp)['Poster'], "posters/%s.jpg" % key)
    except URLError, e:
        print 'No kittez. Got an error code:', e
