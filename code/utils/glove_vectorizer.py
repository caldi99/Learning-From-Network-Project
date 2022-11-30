
import numpy as np
from config import configs


class GloveVectorizer:
    def __init__(self):
        word2vec = {}
        embedding = []
        idx2word = []
        with open(configs.PATH_GLOVE_6B_50D, encoding = configs.ENCODING_TYPE) as file:
            for line in file:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype = configs.FLOAT_32)
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)
        
        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v:k for k,v in enumerate(idx2word)}
        self.V, self.D = self.embedding.shape



    def transform(self, data):
        """
            This function given phrases to convert, it transform it to vectors
            data : 
                phrases to convert
            return :
                vectors that correspond to those phrases
        """
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1        
        return X

