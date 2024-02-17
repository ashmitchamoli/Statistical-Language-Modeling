# How to run
Run all files from the root directory
## tokenizer.py
```bash
python tokenizer.py
```
After running the file, you will be prompted to enter text.

## language_model.py
```bash
python language_model.py <smoothing_type> <corpus_path>
```
After running, you will be prompted to enter a sentence and it's probability score will be provided.

Example usage:
```bash
python language_model.py g corpus.txt
```

## generator.py
```bash
python generator.py <smoothing_type> <corpus_path> <k>
```
After running, you will be prompted to enter a prompt for generation.

NOTE -: Enter a prompt containing more than 1 token.

Example usage:
```bash
python generator.py i corpus.txt 3
```

# Source Code
The source code for tokenization and preprocessing can be found in the `language_modeling/preprocessing` folder. `Tokenizer.py` contains the class for tokenization. The file `language_modeling/models/NGramModel.py` contains the class for nGram language model.

# Experimentation
## Generation
### Normal Smoothing
Sampling scheme: pick the highest probablity word.

On normal smoothing, the generation of the model varies greatly with the value of N. For N=2, the model generates the same sentence over and over again. For N=3, the model generates some something that is not gibberish but starts repetition after a few tokens. For N=4, generates good sentences, which oftentimes do not make semantic sense. For N=5, the model generates good and fluent text.

Output for N=5: "I am a man misunderstood. I am being made a scapegoat of. I am a struggler now at the end of the ballad. Und alle Schiffe br ü cken. The driver never said a word, good, bad or indifferent, but merely watched the two figures, as he sat on his lowbacked car, both black, one full, one lean, walk towards the railway bridge, _ to be married by Father Maher _."

### GT Smoothing
Sampling Scheme: Randomly sample from the probability distribution over the entire vocabulary.
This is done because for higher n-gram models (N > 4), the frequency of seen n-grams becomes so low that the probability of these n-grams is negligible as compared to n-grams which do not appear in the corpus. As a result, the model essentially always picks a random word from the vocabulary with uniform probability.

It is observed that as the value of N is increased, not a lot of improvement is seen in terms of fluency. However, this fluency is short ranged and only some phrases make sense.

### Linear Interpolation
Sampling Scheme: Randomly sample from the probability distribution over the entire vocabulary.

As the value of N increases, there is a clear improvement in fluency of the language model unlike in the good turing case. However the responses still do not make sense semantically but are only grammarically correct.