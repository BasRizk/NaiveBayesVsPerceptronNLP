"""
NaiveBayes is a generative classifier based on the Naive assumption that features are independent from each other
P(w1, w2, ..., wn|y) = P(w1|y) P(w2|y) ... P(wn|y)
Thus argmax_{y} (P(y|w1,w2, ... wn)) can be modeled as argmax_{y} P(w1|y) P(w2|y) ... P(wn|y) P(y) using Bayes Rule
and P(w1, w2, ... ,wn) is constant with respect to argmax_{y} 
Please refer to lecture notes Chapter 4 for more details
"""

from collections import Counter, defaultdict
from math import log
import operator

import numpy as np
from Features import Features, tokenize
from Model import *


class NBFeatures(Features):
    @classmethod 
    def get_features(cls, tokenized, model):
        features = []
        token_to_embed = model['token_to_embed']
        for token in tokenized:
            embed = token_to_embed.get(token)
            if embed is not None:
                features.append(embed)
            else:
                features.append(token_to_embed['__OOV__'])
        return features

class NaiveBayes(Model):
    
    def __init__(self, model_file, vocab_size=None):
        super().__init__(model_file)
        self.vocab_size = vocab_size
        
        
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        
        wprobdenom = '__ALL__'
        
        nbFeatures = NBFeatures(input_file, vocab_size=self.vocab_size)
        
        model = {
            'type': NaiveBayes.__class__,
            'categories_probs': {},
            'words_probs': {},
            'options': nbFeatures.labelset,
            'token_to_embed': nbFeatures.token_to_embed,
            'embed_to_token': nbFeatures.embed_to_token,
            'vocab_size': self.vocab_size,

            # 'label_to_embed': nbFeatures.label_to_embed,
            # 'embed_to_label': nbFeatures.embed_to_label,
        }
                
        wscores = defaultdict(lambda: Counter())
        cscores = Counter()
        
        features_list = list(map(lambda x: NBFeatures.get_features(x, model), nbFeatures.tokenized_text))
        # Y_true = list(map(lambda x: model['label_to_embed'][x], nbFeatures.labels))
        
        cutoff = int(len(features_list)*0.9)
        X_train, X_valid = features_list[:cutoff], features_list[cutoff:]
        Y_train, Y_valid = nbFeatures.labels[:cutoff], nbFeatures.labels[cutoff:]
        
        for features, label in zip(X_train, Y_train):
            cscores[label] += 1
            for f in features:
                wscores[label][f] += 1
                wscores[label][wprobdenom] += 1
        
        # Laplace Smoothing (+1)
        for label in model['options']:
            wprob = {}
            for token in nbFeatures.token_to_embed:
                embed = model['token_to_embed'][token]
                wprob[embed] = 1 / (wscores[label][wprobdenom] + 1)
            model['words_probs'][label] = wprob
        
        for label in model['options']:
            model['categories_probs'][label] =\
                cscores[label] / len(features)
            for feature, score in wscores[label].items():
                # Laplace Smoothing (+1)
                # Overriding vocab values if applicable
                model['words_probs'][label][feature] = (score + 1) / (wscores[label][wprobdenom] + 1)
        
        
        # Validate
        train_err =\
            np.sum(np.array(self._classify(X_train, model)) != np.array(Y_train))/len(Y_train)

        valid_err =\
            np.sum(np.array(self._classify(X_valid, model)) != np.array(Y_valid))/len(Y_valid)
            
        print(f'TrainErr = {train_err}, ValidErr = {valid_err}', end='\n')
            
        ## Save the model
        self.save_model(model)
        print('Saved model.')
        return model


    def _classify(self, features_list, model):
        def evaluate(features, option, model):
            score = log(model['categories_probs'][option])
            for f in features:
                score += log(model['words_probs'][option][f])
            return score    
        
        preds = []
        for features in features_list:
            scores = {}
            for option in model['options']:
                scores[option] = evaluate(features, option, model)
            preds.append(
                max(scores.items(), key=operator.itemgetter(1))[0]
            )
        return preds
        
    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """         
        with open(input_file) as file:
            tokenized_sentences =\
                map(tokenize, file.read().splitlines())

        features_list = list(map(lambda x: NBFeatures.get_features(x, model), tokenized_sentences))
        preds = self._classify(features_list, model)    
        return preds


