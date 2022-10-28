## SMS Spam Collection Detection

### Data and Packages/skills used -
**Data** : [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

**Programming Language** : Python

**Packages** : pandas, numpy, textwrap, matplotlib, sklearn, spaCy, BeautifulSoup, XGBoost
**Data Science skills** - NLP, Sparse Embeddings TF-IDF vectors, Manual Feature Engineering

### EDA -

86% is ham and 14% is spam.

![Spam and Ham count](https://user-images.githubusercontent.com/77465643/198736617-fdbaf228-e7c8-450d-b43a-e4c9c996aae3.png)

### Evaluation Metric - 0.5F
In this case we consider 'spam' as positive label and 'ham' as negative label since the goal is to predict spam emails correctly. If spam emails remain undetected (Type II error /FN) it is still okay. But if we miss any critical email because it is classified as 'spam' (Type I error/FP) it will be problematic. Usually, spam folders are neglected so this might impact the user. Hence, here precision is more important i.e. out of all the spam emails predicted what percentage is truly spam emails. However, out dataset is highly imbalanced with higher count for the negative class (i.e. ham). The likeliness of predicting the negative label for new observation is more than the positive label. This means, we also need to reduce False Negatives (Type II error) to predict the class correctly. The F measure gives the same weightage to both precision and recall.Thus, we use Fbeta measure with beta = 0.5. This is a useful metric to use when both precision and recall are important but slightly more attention is needed on one or the other, in our case false positives are more important than false negatives. The smaller beta value gives more weightage to precision and less to recall. Hence, we use 'F0.5 score' as a metric for evaluation of the model.

### Classification Model Building -

#### Pipeline1. Data Processing + Sparse Embeddings (TF-IDF) + ML Model
Attached custom preprocessor is used.
![Pipeline1](https://user-images.githubusercontent.com/77465643/198736652-2ec898c3-6b3b-4c3e-817b-63d1feb4a562.png)

#### Pipeline2. Data Processing + Manual Features + ML Model

Manual features are extracted and used these as the input to our XGBoost Classification model.

![Pipeline2](https://user-images.githubusercontent.com/77465643/198736679-e98c3de4-9de6-42c9-8add-115f4d8ffd74.png)

#### Pipeline3: Data Processing + Combine Manual Features and TfID vectors + ML Model
![Pipeline3](https://user-images.githubusercontent.com/77465643/198736708-5a09cb35-6f10-4952-beeb-e25f4fff64ec.png)

### Conclusion - 
* Analyzed spam data and developed three pipelines using TF-IDF vectors, manual features, and combined TF-IDF vectors. The evaluation metric used was
F0.5 giving slightly more importance to precision than recall for this imbalanced dataset.
* Achieved a maximum F0.5 score of **0.9458** using an **XGBoost Classifier** which turned out to be the best model.
