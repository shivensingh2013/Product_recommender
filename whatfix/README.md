# ProdSearch
Personalized Product Search with Product Reviews

## Installation 

1) Follow the steps given in script "assignment_code\scripts\extract_data.sh" to download and process dataset of Amazon product search

2) Install the libraries required using -
   ''' pip install requirement.txt'''

## Dataset:
For each user, we sort his/her purchased items by time and divide items to train/validation/test in a chronological order. 

The complete data processing has been done as suggested in data preparation steps of (https://github.com/QingyaoAi/Explainable-Product-Search-with-a-Dynamic-Relation-Embedding-Model) 

1- Folder "amazon_data" consists of the original Amazon data
2- Folder "amazon_data_processed" consists of processing done by the script - 
''' assignment_code\scripts\extract_data.sh '''

Note: the processing steps has been performed for only 1 core out of 5 cores - "Cell_phone" category


## Train TEM model [1]
To train a transformer-based embedding model (TEM) [1], run 

Steps - 
1) Setup the values in config file containing paramaters for the model
'''.\assignment_code\scripts\config.py'''
2)  python .\assignment_code\scripts\train.py --config .\old\ProdSearch\scripts\config.yaml


## Train RTM model [2]
If you want to run a review-based transformer model (RTM) [2], simply use a different model_name in config
```
model_name=  review_transformer
```

## Testing the model 

'''python .\assignment_code\scripts\inference.py --config .\old\ProdSearch\scripts\config.yaml
'''

## References
[1] Keping Bi, Qingyao Ai, W. Bruce Croft. A Transformer-based Embedding Model for Personalized Product Search. In Proceedings of SIGIR'20.

[2] Keping Bi, Qingyao Ai, W. Bruce Croft. Learning a Fine-Grained Review-based Transformer Model for Personalized Product Search. In Proceedings of SIGIR'21.
