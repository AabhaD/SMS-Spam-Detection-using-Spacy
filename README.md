## SMS Spam Collection Detection

### Data and Packages/skills used -
**Data** : [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

**Programming Language** : Python

**Packages** : pandas, numpy, matplotlib, seaborn, sklearn
Data Science skills - NLP, Sparse Embeddings TF-IDF vectors, Manual Feature Engineering, XGBoost

import pandas as pd
import numpy as np
from pathlib import Path
import textwrap as tw
import matplotlib.pyplot as plt

# learning Curves
from sklearn.model_selection import learning_curve

# save and load models
import joblib

import re
from bs4 import BeautifulSoup
import spacy

#from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from  sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support



### Data cleaning -

### EDA -

### Model Building -

### Model Performance -

### Conclusion - 
* Analyzed spam data and developed three pipelines using TF-IDF vectors, manual features, and combined TF-IDF vectors. The evaluation metric used was
F0.5 giving slightly more importance to precision than recall for this imbalanced dataset.
* Achieved a maximum F0.5 score of **0.9458** using an **XGBoost Classifier** which turned out to be the best model.
