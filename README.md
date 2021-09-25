# TS_Control_Factual_Consistency

## Prerequisites

Linux with python 3.6 or above (not compatible with python 3.9 yet).
## Installing
```
git clone git@github.com:pelican9/TS_Control_Factual_Consistency.git
cd muss/
pip install -e .  # Install package
python -m spacy download en_core_web_md fr_core_news_md es_core_news_md  # Install required spacy models
```

Becasue the size of these folders are huge, they need to be downloaded from the following link and added under /muss: (https://drive.google.com/drive/folders/1taEhdNIRB_BVLHf7QaCWqWZ4303rg1or?usp=sharing)

muss/resources includes BPE dictionary and datasets with different number of words in prefix. muss/qualitative includes saved dataframe e.g. named entities for each sentence. muss/experiments includes the all trained models.

## Train
Download the folder bart_bpe, and create your dataset in muss/resources/datasets. And change the dataset name in **train.ipynb**.

## Generate output
If you only want to do simplification on custom sentence, then download the model from muss/experiments is enough for generation. The look up table for model is shown below.
| Operation                          | Model                          | Model ID | Experiment ID|
|------------------------------------|--------------------------------|----------|----------------|
| Preserving                         | complex/complex                | 18       |local_1629593348299
| Preserving                         | complex/simple                 | 19       |local_1629593348299
| Preserving                         | simple/complex                 | 20       |local_1629593322552
| Preserving                         | simple/simple                  | 21       |local_1629593322552
| Preserving                         | both/complex                   | 22       |local_1629750798219
| Preserving                         | both/simple                    | 23       |local_1629750798219
| Preserving                         | both/both                      | 39       |local_1629750798219
| Lexical Simplification             | all pairs                      | 37       |local_1631390882572
| Lexical Simplification             | filtered pairs                 | 38       |local_1631390826128
| Preserving+ Lexical Simplification | simple/simple/filtered   pairs | 40       |local_1631567843618


To simplify custom sentence, first change the input file depends on what operation is conducted: (NE for Named Entity)
```
if muss_output: 
  # the input sentence has no prefix
  complex_file_dir = '/content/drive/MyDrive/muss/scripts/contract_no_token.en'
  
elif NE_output: 
  # the input sentence has named entities in prefix
  complex_file_dir = '/content/drive/MyDrive/muss/scripts/contract_NE_token.en'
  
elif CERF_output: 
  # the input sentence has hard words in prefix
  complex_file_dir = '/content/drive/MyDrive/muss/scripts/contract_ABCD_token.en'
  
elif NE_CERF_output: 
  # the input sentence has named entities and hard words in prefix
  complex_file_dir = '/content/drive/MyDrive/muss/scripts/contract_NE_ABCD_token.en'
```        
Then define the dictionary that includes model path and the corresponding test set path in **generate.ipynb**, then use function *generate_output* with sample=True and, say operation preserving is carried out, NE_output=True, to generate outputs.

To simplify ASSET test set, first define the dictionary that includes model path and the corresponding test set path in **generate.ipynb**, then use function *generate_output* to generate outputs.


