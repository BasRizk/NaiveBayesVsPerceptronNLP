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

from Features import Features, tokenize
from Model import *


class NaiveBayes(Model):
        
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        
        wprobdenom = '__ALL__'
        
        features = Features(input_file)
        
        model = {
            'type': NaiveBayes.__class__,
            'categories_probs': {},
            'words_probs': {},
            'options': features.labelset
        }
                
        wscores = defaultdict(lambda: Counter())
        cscores = Counter()
        
        for tokens, label in zip(features.tokenized_text, features.labels):
            cscores[label] += 1
            for token in tokens:
                wscores[label][token] += 1
                wscores[label][wprobdenom] += 1
        
        # Laplace Smoothing (+1)
        for label in features.labelset:
            wprob = {}
            for token in features.tokens_count:
                wprob[token] = 1 / (wscores[label][wprobdenom] + 1)
            model['words_probs'][label] = wprob
        
        for label in features.labelset:
            model['categories_probs'][label] =\
                cscores[label] / len(features.tokenized_text)
            for token, score in wscores[label].items():
                # Laplace Smoothing (+1)
                # Overriding vocab values if applicable
                model['words_probs'][label][token] = (score + 1) / (wscores[label][wprobdenom] + 1)
            
        ## Save the model
        self.save_model(model)
        return model

    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """ 
        
        def evaluate(tokens, option, model):
            score = log(model['categories_probs'][option])
            for token in tokens:
                # breakpoint()
                score += log(model['words_probs'][option][token])
            return score    
        
        with open(input_file) as file:
            tokenized_sentences =\
                map(tokenize, file.read().splitlines())
        
        # model = self.load_model()
        preds = []
        for tokens in tokenized_sentences:
            scores = {}
            for option in model['options']:
                scores[option] = evaluate(tokens, option, model)
            preds.append(
                max(scores.items(), key=operator.itemgetter(1))[0]
            )
                    
        return preds


