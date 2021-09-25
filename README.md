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

Becasue some folders are huge, they needs to be downloaded from the following link and folders needs to be added under /muss: (https://drive.google.com/drive/folders/1taEhdNIRB_BVLHf7QaCWqWZ4303rg1or?usp=sharing)

muss/resources includes BPE dictionary and datasets. muss/qualitative includes saved dataframe e.g. named entities for each sentence. muss/experiments includes the all trained models.

 
## Generate output
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
Then define the dictionary that includes model path and the corresponding test set path in **generate.ipynb**, then use function generate_output with sample=True and, say operation preserving is carried out, NE_output=True, to generate outputs.

To simplify ASSET test set, first define the dictionary that includes model path and the corresponding test set path in **generate.ipynb**, then use function generate_output to generate outputs.

The look up table for model is shown as below.
| Operation                          | Model                          | Model ID |
|------------------------------------|--------------------------------|----------|
| Preserving                         | complex/complex                | 18       |
| Preserving                         | complex/simple                 | 19       |
| Preserving                         | simple/complex                 | 20       |
| Preserving                         | simple/simple                  | 21       |
| Preserving                         | both/complex                   | 22       |
| Preserving                         | both/simple                    | 23       |
| Preserving                         | both/both                      | 39       |
| Lexical Simplification             | all pairs                      | 37       |
| Lexical Simplification             | filtered pairs                 | 38       |
| Preserving+ Lexical Simplification | simple/simple/filtered   pairs | 40       |


