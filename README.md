# Multidocsum
Multi-document summarization for Vietnamese news.

# Installation
```shell
# Clone this repo
git clone https://github.com/phuocvtran/multidocsum.git
# Install multidocsum and all its dependencies
cd multidocsum
pip install .
# Download & preprocess ViMs dataset
python setup_data.py
```
# Usage
```python
# Import MEAD
from src.models.extracting import MEAD


# Defining parameters
params = {
    # File of stop words, newline seperated
    'stop_words_path': '../data/external/stopwords.txt', 
    # All punctuation on a single line e.g. ?!.,"'“”-();:[]/
    'punc_path': '../data/external/punctuation.txt', 
    # Remove redundant sentences from the summary?
    'remove_redundant_sent': False,
    # Weight of centroid score
    'wc': 1.0, 
    # Weight of position score
    'wp': 1.0, 
    # Weight of first-sentence overlap score
    'wf': 1.0, 
    # Number of centroid to be extracted
    'n_centroid': 10, 
    # Max score of position score
    'C_max': 10, 
    # N-gram range & max length of TF*IDF vector for calculate first-sentence overlap score
    # See sklearn.feature_extraction.text.TfidfVectorizer
    'ngram_range': (1, 1), 
    'max_features': 100,
    # Function to calculate first-sentence overlap score
    # take two numpy array and return a number
    'func': np.dot
}

# Initialize MEAD
summerizer = MEAD(**params)

# data_path is path to folder that contain all the documents in .txt format
# e.g.
#   Cluster_001
#   |_ news01.txt
#   |_ news02.txt
#   |_ ...
#   |_ a_lot_of_news.txt
#
# output is a list of extracted sentences.
summary = summarier.extract(data_path='../data/interim/extract_content/Cluster_001', 
                            compression_rate=0.05,
                            save_as=None)
```
See in this [notebook](notebooks/02-tvp-sample-usage.ipynb).

# References
[Radev, D. R., Jing, H., Styś, M., & Tam, D. (2004). Centroid-based summarization of multiple documents.](https://dollar.biz.uiowa.edu/~street/radev04.pdf)
