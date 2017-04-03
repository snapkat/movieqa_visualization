import MovieQA
import copy
import datetime
import logging
import math
import nltk
import numpy as np
from util import log_time_info
from collections import Counter

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


class QA(object):

    def __init__(self, _imdb_key, _q, _ans, _correct_index):
        self.question = _q
        self.imdb_key = _imdb_key
        self.answers = _ans
        self.correct_index = _correct_index


class TfIdfTester(object):

    def __init__(self, plots, qas):
        self.plots, self.qas = (plots, qas)

        self.tokenizer = RegexpTokenizer("[\w]+")
        self.lemmatizer = WordNetLemmatizer()

        self.qa = []
        self._process_text()

        # --- Per plot unique word counts. ---
        self.total_word_counts = Counter()

        for imdb_key in self.plots:
            p_ctr = Counter()
            for counter in self.plots[imdb_key]:
                p_ctr += Counter(counter)
            self.total_word_counts += Counter(p_ctr.keys())

        for qa in self.qa:
            p_ctr = Counter()
            for counter in qa.answers:
                p_ctr += Counter(counter)
            p_ctr += Counter(qa.question)
            self.total_word_counts += Counter(p_ctr.keys())

        self.total_words = sum(self.total_word_counts.values()) + 0.0

        print 'TFIDF Setup Finished'
        # Story becomes list of counters.

    def _process_sentence(self, sentence):
        ''' Returns dictionary of tf values of each word in sentence.
            tf(word): #occurences(word)/#total_words
        '''
        words = self.tokenizer.tokenize(sentence)
        word_list = [self.lemmatizer.lemmatize(w.lower()) for w in words]
        total = len(word_list) + 0.0
        word_count = Counter(word_list)
        word_count = dict(word_count)
        return {w: word_count[w] / total for w in word_count}

    def _process_text(self):
        ''' Calculates TF values for each document (sentence).
        '''
        # Process Plots.
        for imdb_key in self.plots:
            story = []
            for sentence in self.plots[imdb_key]:
                story.append(self._process_sentence(sentence))
            self.plots[imdb_key] = story

        # Process QAs
        for i, qa in enumerate(self.qas):
            q = self._process_sentence(qa.question)

            processed_answers = []
            for ans in qa.answers:
                processed_answers.append(self._process_sentence(ans))

            self.qa.append(QA(qa.imdb_key, q, processed_answers,
                              qa.correct_index))

    def tfidf(self, n_qa):
        qa = self.qa[n_qa]
        plot = self.plots[qa.imdb_key]
        corpus_tf = plot + qa.answers + [qa.question]

        vocab = set()
        doc_freq = Counter()
        for count in corpus_tf:
            vocab = set(count.keys()) | vocab
            doc_freq += Counter(count.keys())
        vocab = list(vocab)
        n_docs = len(corpus_tf)

        # Build TF
        tf = np.zeros((n_docs, len(vocab)))
        for doc_id in range(n_docs):
            for idx, word in enumerate(vocab):
                if word in corpus_tf[doc_id]:
                    tf[doc_id, idx] = corpus_tf[doc_id][word]

        # Build IDF
        idf = np.zeros((1, len(vocab)))
        for idx, word in enumerate(vocab):
            if word in self.total_word_counts:  # 0 if word not in plot -_-
                idf[0, idx] = np.log((n_docs) / (0.0 + doc_freq[word])) 
                idf[0, idx] *= np.log(self.total_words / self.total_word_counts[word]) #Tweak with idf for all docs

        tf_idf = self.normalize(tf * idf)

        vocab_tfidf = []
        for idx, word in enumerate(vocab):
            tf_idf

    def tfidf_weights(self, imdb_key):
        corpus_tf = self.plots[imdb_key]

        vocab = set()
        doc_freq = Counter()
        for count in corpus_tf:
            vocab = set(count.keys()) | vocab
            doc_freq += Counter(count.keys())
        vocab = list(vocab)
        n_docs = len(corpus_tf)

        # Build TF
        tf = np.zeros((n_docs, len(vocab)))
        for doc_id in range(n_docs):
            for idx, word in enumerate(vocab):
                if word in corpus_tf[doc_id]:
                    tf[doc_id, idx] = corpus_tf[doc_id][word]

        # Build IDF
        idf = np.zeros((1, len(vocab)))
        for idx, word in enumerate(vocab):
            if word in self.total_word_counts:  # 0 if word not in plot -_-
                idf[0, idx] = np.log((n_docs) / (0.0 + doc_freq[word])) 
                idf[0, idx] *= np.log(self.total_words / self.total_word_counts[word]) #Tweak with idf for all docs

        tf_idf = self.normalize(tf * idf)

        return tf_idf

    def normalize(self, mat):
        if mat.ndim == 1:
            return mat / (np.linalg.norm(mat) + 1e-6)
        return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-6)

    def score(self, n_qa):
        tf_idf = self.tfidf(n_qa)
        imdb_key = self.qa[n_qa].imdb_key
        plot_matrix = tf_idf[:len(self.plots[imdb_key])]
        question_vec = tf_idf[-1]
        answer_matrix = tf_idf[len(self.plots[imdb_key]): -1]

        # Similarity between sentences.
        qscore = plot_matrix.dot(question_vec).reshape(-1, 1)
        ascore = plot_matrix.dot(answer_matrix.T)

        return ascore, qscore

    def predict_tfidf(self, n_qa):
        ''' Returns chosen answer, reference plot line #, and max score.
        '''
        ascore, qscore = self.score(n_qa)
        score = ascore*qscore
        prediction = np.unravel_index(score.argmax(), score.shape)
        return prediction[1], prediction[0], score[prediction]

        #return ascore[qscore.argmax()].argmax(), 0, 0

    @log_time_info
    def test(self):
        ''' Run w2v on the plots.
        '''

        # Start testing.
        nCorrect = 0.0
        nCorrectSame = 0.0
        prediction_distribution = [0, 0, 0, 0, 0]
        correct_distribution = [0, 0, 0, 0, 0]
        nTried = len(self.qa)

        for n_qa, q in enumerate(self.qa):
            prediction = self.predict_tfidf(n_qa)
            if prediction[0] == q.correct_index:
                nCorrect += 1
            prediction_distribution[prediction[0]] += 1
            correct_distribution[q.correct_index] += 1

        print("TFIDF Accuracy:", nCorrect / nTried)
        print("Predicted Answers: ", prediction_distribution)
        print("Correct Answers: ", correct_distribution)
        return(nCorrect / nTried)


if __name__ == "__main__":

    extension = '[Sentence Level Documents]'
    print "Testing TFIDF with... %s" % extension
    mqa = MovieQA.DataLoader()
    print "-----Training Data-----"
    story_raw, qa = mqa.get_story_qa_data('train', 'plot')
    tfidf_tester = TfIdfTester(story_raw, qa)
    tfidf_tester.test()

    print "-----Validation Data-----"
    story_raw, qa = mqa.get_story_qa_data('val', 'plot')
    tfidf_tester = TfIdfTester(story_raw, qa)
    tfidf_tester.test()
