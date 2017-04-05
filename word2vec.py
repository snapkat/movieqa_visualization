import numpy as np
import pickle

class Word2Vec(object):
    def __init__(self, extension='plt', postprocess=False):
        # Filenames & Constants
        ID_MAP_FILE = "data/id_map_%s.pkl" % extension
        EMBED_FILE = "data/embed_%s.npz" % extension
        self.SYMBOLS_TO_REMOVE = '"#$%&()*+,/:;<=>@[\]^_`{|}~-' + "'?!"

        # Get word2vec embeddings
        # This is the embeddings that are trained,
        # you access it using the given functions and the given IDs
        self.embeddings = [] # (numWords * embeddingDimensions)
        with np.load(EMBED_FILE) as f:
            self.embeddings = f['embed']
            print('Embeddings loaded')

        # Get the word to word-id mappings
        # Basically, this is a map and isn't train
        self.id_map = self.load_obj(ID_MAP_FILE) # (numWords *  embeddingDimensions)
        self.word_map = dict((v,k) for k,v in self.id_map.iteritems())

        if postprocess:
            self._postprocess()
            print "Self embedding size and values"
            print(self.embeddings.shape)
            print("Self ID Map") # A dictionary
            print(len(self.id_map))

    def n_nearest(self, word, n=6):
        q_idx = self.id_map[word]
        weights = np.dot(self.embeddings, self.embeddings[q_idx])
        top_n = list(np.argsort(weights)[-n-1:])
        top_n.reverse()
        for i in top_n:
            if i == q_idx:
                top_n.remove(q_idx)
        top_n_words = [self.word_map[i] for i in top_n[:n]]
        return top_n_words

    def getNumVocabulary(self):
        return len(self.id_map)

    def load_obj(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def normalize(self, mat):
        if mat.ndim == 1:
            return mat / (np.linalg.norm(mat) + 1e-6)
        return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-6)

    def tokenize_text(self, words):
        ''' Convert cleaned text into list of tokens/word-ids.
        Ret: list of ints (ids)
        '''
        if isinstance(words, str):
            words = words.split()
        tokens = []
        for i, word in enumerate(words):
            id = 0  # 0 is UNK id
            word = word.lower()
            if word in self.id_map:
                id = self.id_map[word]
            tokens.append(id)

        return np.array(tokens)

    def tokenized_as(self, word):
        if word.lower() in self.id_map:
            return word.lower()
        return 'UNK'

    def _clean_text(self, text_raw):
        ''' Splits sentences and then filters.
        '''
        text = text_raw
        if isinstance(text_raw, list):
            text = ' '.join(sentence for sentence in text_raw)
        text = filter(lambda x: x not in self.SYMBOLS_TO_REMOVE, text)
        text = text.split(". ")
        return text

    def _filter_sentence(self, sentence_raw):
        ''' Filters out unused symbols including period.
        '''
        text = filter(lambda x: x not in (
            self.SYMBOLS_TO_REMOVE + '.'), sentence_raw)
        return text

    def _postprocess(self):
        ''' From inarXiv:1702.01417v1 Paper
        '''
        from sklearn.decomposition import PCA
        # Remove non-zero mean.
        vocab_size, n_dim = self.embeddings.shape
        mu = np.sum(self.embeddings, axis=0, keepdims=True) / vocab_size
        self.embeddings = self.embeddings - mu

        # PCA
        D = int(n_dim / 100)
        pca = PCA(n_components=D)
        pca.fit(self.embeddings)

        # Find dominating directions.
        u = np.matmul(self.embeddings, pca.components_.T)
        dom_directions = np.matmul(u, pca.components_)
        self.embeddings = self.embeddings - dom_directions

        print "Finished postprocessing."

    def clean_words(self, words):
        w = []
        for i in range(len(words)):
            w.append(filter(lambda x: x not in self.SYMBOLS_TO_REMOVE +'.', words[i]))
        return w

    def get_raw_word_embeddings(self,words):
        words = self.clean_words(words)
        return self.get_word_embeddings(words)

    def get_word_embeddings(self, words):
        word_ids = self.tokenize_text(words)
        embeddings = self.embeddings[word_ids]
        return embeddings

    def get_sentence_vector(self, sentence):
        clean_sentence = self._filter_sentence(sentence)
        sentence_vector = self.embeddings[self.tokenize_text(clean_sentence)]
        normalized_sentence_vector = self.normalize(
            np.average(sentence_vector, axis=0))
        # print(normalized_sentence_vector.shape)
        return normalized_sentence_vector

    def get_vectors_for_raw_text(self, text):
        ''' Get a matrix of embeddings for a text with multiple sentences (i.e. plot).
        '''
        cleaned_text = self._clean_text(text)
        return self.get_text_vectors(cleaned_text)

    def get_text_vectors(self, cleaned_text):
        ''' Get a matrix of embeddings for a text with multiple sentences (i.e. plot).
            Assume text has already been cleaned.
        '''
        embedding_matrix = np.array([self.get_sentence_vector(line)
                                     if line != ""
                                     else np.zeros(self.embeddings.shape[1])
                                     for line in cleaned_text])

        return embedding_matrix
