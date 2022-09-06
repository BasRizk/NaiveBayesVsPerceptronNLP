""" 
    Basic feature extractor
"""
from operator import methodcaller
import string 
from collections import Counter
import numpy as np

def tokenize(text):
    # TODO customize to your needs
    
    # Remove punctuations (replace with ' ')
    text = text.translate(str.maketrans({key: " ".format(key) for key in string.punctuation}))
    return text.split()

class Features:

    def __init__(self, data_file):
        with open(data_file) as file:
            data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, self.labels = map(list, zip(*data_split))

        self.tokenized_text = [tokenize(text) for text in texts]
        breakpoint()
        
        self.tokens_count =\
            Counter([
                t 
                for tokenized_sentence in self.tokenized_text
                for t in tokenized_sentence
            ])
            
        self.labelset = list(set(self.labels))

        self._make_encoding_dict()
        

    def _make_encoding_dict(self):
        self.token_to_embed = {}        
        self.token_to_embed['__OOV__'] = 0
        self.token_to_embed['__PAD__'] = 1
        
        for i, token in enumerate(self.tokens_count):
            self.token_to_embed[token] = i+2

        self.embed_to_token = {
            embed: token 
            for token, embed in self.token_to_embed.items()
        }       
        
        self.label_to_embed = {}
        self.embed_to_label = {}
        for i, label in enumerate(self.labelset):
            self.label_to_embed[label] = i
            self.embed_to_label[i] = label
            
                    
    @classmethod 
    def get_features(cls, tokenized, model):
        # TODO: implement this method by implementing different classes for different features 
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features 
        
        # TODO check tokenized is 2d array or 1d array (sentence(s))

        features = []
        token_to_embed = model['token_to_embed']
        for sentence in tokenized:
            embed_sentence = []
            for token in sentence:
                embed = token_to_embed.get(token)
                if embed is not None:
                    embed_sentence.append(embed)
                else:
                    embed_sentence.append(token_to_embed['<OOV>'])
            features.append(embed_sentence)
        return features
        # elif model['type'] == Perceptron.__class__:
        #     # TODO
        #     pass
        # else:
        #     pass
                    
                    