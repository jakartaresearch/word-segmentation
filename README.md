# Indonesian Word Segmentation
---
IWS is a research project that focuses on word segmentation in Bahasa.
According to Wikipedia, word segmentation is also known as text segmentation is the process of separating the text into meaningful units, such as Words, Sentence, or Topics. The problem is non-trivial because while some written languages have explicit word boundary markers, such as the word spaces of written English and the distinctive initial, medial and final letter shapes of Arabic, such signals are sometimes ambiguous and not present in all written languages.

Bahasa has a straight boundary between words, but we found that some people write unconsciously for their writtings. Therefore, this research aims to help writers for better writings.

Why IWS:
IWS
- focus on seperating words without dictionary
- Enables you to correct indonesian text with easy load

Project Trees:


    |--data 
        |--raw
        |--clean
    |--notebook
    |--reports
    |--models
    |--requirements.txt
    |--README.md
    
    
## Instalation
---
**Instalation Requirements**:
    
    pip install -r requirements.txt
    
## Dataset
The dataset can be found [here](https://drive.google.com/drive/folders/1zRq6e9ndUOX7v6Qz-EU9ECSfxSTJsv6u?usp=sharing)

## Embedding
The embedding is using gensim word2vec based on the "data_clean_100k.res" as the corpus. 
There are multiple length of embedding (25,50 and 75)

## Timeline
Detail timeline can be found [here](https://docs.google.com/spreadsheets/d/170BDuEffhWfKkcdC0TCmQYmuEhOYN0RumVqhaVoHxAU/edit?usp=sharing)