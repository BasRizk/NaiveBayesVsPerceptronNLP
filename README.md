## Instructions for Training
### Parameters:
  - `-m` model type : {`perceptron`, `naivebayes`}
  - `-i` dataset relative filepath
  - `-o` output model relative filepath
  - `-vocab_size` number of values to pick
    - any value between `0` to `1`: fraction of the number of unique tokens available
    - value equals to `1`: means use the whole vocab set
    - value greater than `1`: the size of vocab to use
  - `-epochs`* number of epochs to run the model
  - `-feat`* featurization method:
    - `word_count`
    - `word_presence`
    - `first_best`
    - `last_best`
    - `center_best`

\* Used with Perceptron model only


### Example
> `python train.py -m perceptron -i datasets/questions.train.txt -o perceptron.questions.model -epochs 25 -vocab_size 1 -feat word_presence`

## Instructions for Classifying
### Parameters:
  - `-m` model filename (either start with `nb` or `perceptron`)
  - `-i` test data-set relative filepath
  - `-o` output (inference) desired relative filepath
  
  
### Example
> `python classify.py -m nb.4dim.model -i 4dim.sample.txt -o 4dim.out.txt`