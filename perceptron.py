"""
 Refer to Chapter 5 for more details on how to implement a Perceptron
"""
import numpy as np

from Features import Features, tokenize
from Model import *

class PPFeatures(Features):
    @classmethod 
    def get_features(cls, tokenized_list, model):
        featurization_apply = {
            "word_count": lambda x: PPFeatures.get_features_word_count(x, model),
            "word_presence": lambda x: PPFeatures.get_features_word_count(x, model, feat_word_presence=True),
            "last_best": lambda x: PPFeatures.get_features_acc_order_priority(x, model),
            "first_best": lambda x: PPFeatures.get_features_acc_order_priority(x, model),
            "end_best": lambda x: PPFeatures.get_features_acc_order_priority(x, model),
            "center_best": lambda x: PPFeatures.get_features_acc_order_priority(x, model)
        }
        return featurization_apply[model['featurization_method']](tokenized_list)
    
    @classmethod
    def get_features_word_count(cls, tokenized_sentence, model, feat_word_presence=False):
        token_to_embed = model['token_to_embed']
        features = np.zeros(len(token_to_embed), dtype=np.int8)
            
        for token in tokenized_sentence:
            embed = token_to_embed.get(token)
            if embed is not None:
                features[embed] += 1
            else:
                features[token_to_embed['__OOV__']] += 1

        # TODO test out
        if feat_word_presence:
            features[features >= 1] = 1
        #     print('Featurization Method: Word Presence')
        # else:
        #     print('Featurization Method: Word Count')

        return features
    
    @classmethod
    def get_features_acc_order_priority(cls, tokenized_sentence, model):
        # print('Featurization Method: Last Best')
        token_to_embed = model['token_to_embed']
        features = np.zeros(len(model['token_to_embed']))
        num_of_tokens = len(tokenized_sentence)


        mean = int(num_of_tokens/2) + 1
        variance = np.var(np.arange(1, 20))
        def calc_center_best(i):
            return (np.e**(-0.5*((i-mean)/variance)**2))/(variance*(np.sqrt(2*np.pi)))

            
        priority_calc = {
            'last_best': lambda i: (i+1)/num_of_tokens,
            'first_best': lambda i: (num_of_tokens-(i+1))/num_of_tokens,
            'end_best': lambda i: max(priority_calc['first_best'](i+1), priority_calc['last_best'](i+1)),
            'center_best': lambda i: calc_center_best(i+1)
        }
        
        
     
        # tmp = []
        for i, token in enumerate(tokenized_sentence):
            embed = token_to_embed.get(token, token_to_embed['__OOV__'])
            features[embed] += priority_calc[model['featurization_method']](i)
            # tmp.append(priority_calc[model['featurization_method']](i))
        # breakpoint()
        
        return features
    

        
        
    
class Perceptron(Model):
    # TODO test different vocab_size (all vs with limit)
    def __init__(self, model_file, epochs=20, learning_rate=1,
                 vocab_size=None, featurization_method=None):
        super().__init__(model_file)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.featurization_method = featurization_method
        
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        
        ppFeatures = PPFeatures(input_file, vocab_size=self.vocab_size)
        model = {
            'type': Perceptron.__class__,
            # TODO maybe make something more rigid that is
            # flexible no matter the embedding format
            'weights' : np.random.rand(
                len(ppFeatures.token_to_embed), len(ppFeatures.labelset)
            ),
            'options' : ppFeatures.labelset,
            'token_to_embed': ppFeatures.token_to_embed,
            'embed_to_token': ppFeatures.embed_to_token,
            'label_to_embed': ppFeatures.label_to_embed,
            'embed_to_label': ppFeatures.embed_to_label,
            'vocab_size': self.vocab_size,
            'featurization_method': self.featurization_method
        }
        
        print(f'Featurization Method Applied : {self.featurization_method}')

        features_list = np.array(list(
            map(lambda x: PPFeatures.get_features(x, model), ppFeatures.tokenized_text)
        ))
        
        Y_true = np.array(list(
            map(lambda x: model['label_to_embed'][x], ppFeatures.labels)
        ))
        
        cutoff = int(features_list.shape[0]*0.9)
        X_train, X_valid = features_list[:cutoff], features_list[cutoff:]
        Y_train, Y_valid = Y_true[:cutoff], Y_true[cutoff:]
        
        for epoch in range(self.epochs):
            print(f'Training at Epoch {epoch + 1}/{self.epochs}', end=' ')
            for x, y in zip(X_train, Y_train):
                hypothesis = np.argmax(x@model['weights'])
                if hypothesis != y:
                    model['weights'][:, y] += self.learning_rate*x
                    model['weights'][:, hypothesis] -= self.learning_rate*x
                    
            # Validate
            train_err =\
                np.sum(
                    np.argmax(X_train@model['weights'], axis=1) != Y_train
                )/Y_train.shape[0]
            valid_err =\
                np.sum(
                    np.argmax(X_valid@model['weights'], axis=1) != Y_valid
                )/Y_valid.shape[0]
            print(f'TrainErr = {train_err}, ValidErr = {valid_err}', end='\n')
            
            
        ## Save the model
        self.save_model(model)
        print('Saved model.')
        return model


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
        
        features_list = np.array(list(
                    map(lambda x: PPFeatures.get_features(x, model), tokenized_sentences)
                ))
        preds = []

        embedded_labels = np.argmax(features_list@model['weights'], axis=1)
        for embed_label in embedded_labels:
            preds.append(model['embed_to_label'][embed_label])
                    
        return preds
