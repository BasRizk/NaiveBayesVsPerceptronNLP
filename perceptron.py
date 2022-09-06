"""
 Refer to Chapter 5 for more details on how to implement a Perceptron
"""
import numpy as np

from Features import Features
from Model import *

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
            'weights' : np.random(features_size),
            'options' : features.labelset
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
