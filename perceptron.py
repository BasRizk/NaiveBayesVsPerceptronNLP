"""
 Refer to Chapter 5 for more details on how to implement a Perceptron
"""
import numpy as np

from Features import Features, tokenize
from Model import *

class NBFeatures(Features):
    @classmethod 
    def get_features(cls, tokenized, model):
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

                    
    
class Perceptron(Model):
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        ## TODO write your code here
        
        features = Features(input_file)
        features_size = 0 # TODO
        model = {
            'type': Perceptron.__class__,
            'weights' : np.random(features_size),
            'options' : features.labelset,
            'token_to_embed': features.token_to_embed,
            'embed_to_token': features.embed_to_token
        }
        epochs = 10 # TODO tunable
        for epoch in range(epochs):
            for tokens, label in zip(features.tokenized_text, features.labels):
                hypothesis = None # TODO classify sentence
                if hypothesis != label:
                    pass
                    # TODO !!!
                    model['weights'] +=\
                        features(sentence, label) - features(sentence, hypothesis)
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
        ## TODO write your code here (and change return)
        preds = None
        return preds
