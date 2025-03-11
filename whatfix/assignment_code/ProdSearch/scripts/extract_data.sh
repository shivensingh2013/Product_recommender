#!/bin/bash

# Download dataset
wget -O review.qz http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz
# Download meta data containing category info
wget -O meta.gz http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Cell_Phones_and_Accessories.json.gz

# Remove stop words and preprocess data
java -Xmx4g -jar "./amazon_data_processed/code/utils/AmazonDataset/jar/AmazonReviewData_preprocess.jar" false "C:/Users/IHG6KOR/Desktop/shiv/Portfolio/Job_related/whatfix/processed_data/cell_phone/review.gz" "C:/Users/IHG6KOR/Desktop/shiv/Portfolio/Job_related/whatfix/processed_data/cell_phone/preprocessed_data.gz"

# Indexing the dataset
python "./amazon_data_processed/code/utils/AmazonDataset/index_and_filter_review_file.py" "./cell_phone/preprocessed_data.gz" "./cell_phone/temp_data/" 5

# Matching the meta data with indexed data
java -Xmx16G -jar "./amazon_data_processed/code/utils/AmazonDataset/jar/AmazonMetaData_matching.jar" false "./cell_phone/meta.gz" "./cell_phone/temp_data/min_count5/"

# Splitting training and testing
python "./amazon_data_processed/code/utils/AmazonDataset/sequentially_split_train_test_data.py" "./cell_phone/temp_data/min_count5/" 0.2 0.3
