export FEATURIZATION_METHOD = "word_presence"
python train.py -m perceptron -i datasets/4dim.train.txt -o perceptron.4dim.model -epochs 30 -vocab_size 5000 -feat FEATURIZATION_METHOD
python train.py -m perceptron -i datasets/odiya.train.txt -o perceptron.odiya.model -epochs 30 -vocab_size 5000 -feat FEATURIZATION_METHOD
python train.py -m perceptron -i datasets/products.train.txt -o perceptron.products.model -epochs 30 -vocab_size 5000 -feat FEATURIZATION_METHOD
python train.py -m perceptron -i datasets/questions.train.txt -o perceptron.questions.model -epochs 30 -vocab_size 5000 -feat FEATURIZATION_METHOD