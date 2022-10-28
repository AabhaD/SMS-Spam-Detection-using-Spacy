from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import spacy
import re
import sys
import custom_preprocessor as cp
from pathlib import Path
from spacy.matcher import Matcher

class ManualFeatures(TransformerMixin, BaseEstimator):
    
    def __init__(self, spacy_model, pos_features = True, ner_features = True, count_features = True, punct_features = True, exclam_features = True, free_features = True, tc_features = True, wordexcl_features = True):
        
        self.spacy_model = spacy_model
        self.pos_features = pos_features
        self.ner_features = ner_features
        self.count_features = count_features    
        self.punct_features = punct_features 
        self.exclam_features = exclam_features
        self.free_features = free_features
        self.tc_features = tc_features
        self.wordexcl_features = wordexcl_features

    # Define some helper functions
    def get_pos_features(self, cleaned_text):
        nlp = spacy.load(self.spacy_model)
        noun_count = []
        aux_count = []
        verb_count = []
        adj_count =[]
        disabled = nlp.select_pipes(disable= ['lemmatizer', 'ner'])
        for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=-1):
            nouns = [token.text for token in doc if (token.pos_ in ["NOUN","PROPN"])] 
            auxs =  [token.text for token in doc if (token.pos_ in ["AUX"])] 
            verbs =  [token.text for token in doc if (token.pos_ in ["VERB"])] 
            adjectives =  [token.text for token in doc if (token.pos_ in ["ADJ"])]     
               
            noun_count.append(int(len(nouns)))
            aux_count.append(int(len(auxs)))
            verb_count.append(int(len(verbs)))
            adj_count.append(int(len(adjectives)))
        return np.transpose(np.vstack((noun_count, aux_count, verb_count, adj_count)))
            
      
    def get_ner_features(self, cleaned_text):
        nlp = spacy.load(self.spacy_model)
        count_ner  = []
        disabled = nlp.select_pipes(disable= ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=-1):
            ners = [ent.label_ for ent in doc.ents] 
            count_ner.append(len(ners))
        return np.array(count_ner).reshape(-1,1)  
   
    def get_count_features(self, cleaned_text):
        list_count_words =[]
        list_count_characters =[]
        list_count_digits=[]
        list_count_numbers=[]
        for sent in cleaned_text:
            words = re.sub(r'\d+\s','',sent)
            numbers = re.findall(r'\d+', sent)
            #print(words)
            #print(numbers)

            count_word = len(words.split())
            count_char = len(words)
            count_numbers = len(numbers)
            count_digits = len(''.join(numbers))

            list_count_words.append(count_word)
            list_count_characters.append(count_char)
            list_count_digits.append(count_digits)
            list_count_numbers.append(count_numbers)  
            
        count_features = np.vstack((list_count_words, list_count_characters,
                                    list_count_digits,list_count_numbers ))
        return np.transpose(count_features)
        
    def get_punct_features(self, cleaned_text):
        nlp = spacy.load(self.spacy_model)
        punct_count =[]
        url_count=[]
        disabled = nlp.select_pipes(disable= ['lemmatizer', 'ner'])
        for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=-1):
            puncts = [token.text for token in doc if token.is_punct] 
            urls =  [token.text for token in doc if token.like_url] 
          
            punct_count.append(int(len(puncts)))
            url_count.append(int(len(urls)))
            
        return np.transpose(np.vstack((punct_count, url_count)))
      
    def get_exclam_features(self, cleaned_text):
        nlp = spacy.load(self.spacy_model)
        matcher = Matcher(nlp.vocab)
        exclam_count =[]
        disabled = nlp.select_pipes(disable= ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=-1):
            matcher.add("exclamation",[[{"ORTH": {"REGEX": "\!"}}]])
            matches = matcher(doc)
            exclam_count.append(len([doc[start:end].text for match_id, start, end in matches]))

        return np.array(exclam_count).reshape(-1,1)

    def get_free_features(self, cleaned_text):
        nlp = spacy.load(self.spacy_model)
        matcher = Matcher(nlp.vocab)
        free_count =[]
        disabled = nlp.select_pipes(disable= ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=-1):
            matcher.add("free",[[{"LOWER":"free"}]])
            matches = matcher(doc)     
            free_count.append(len([doc[start:end].text for match_id, start, end in matches]))
           
        return np.array(free_count).reshape(-1,1)

    def get_tc_features(self, cleaned_text):
        nlp = spacy.load(self.spacy_model)
        matcher = Matcher(nlp.vocab)
        tc_count =[]
        disabled = nlp.select_pipes(disable= ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=-1):
            matcher.add("tc",[[{"ORTH": {"REGEX": "(T|t)s?(&|and)(C|c)s?"}}] ])
            matches = matcher(doc)         
            tc_count.append(len([doc[start:end].text for match_id, start, end in matches]))
            
        return np.array(tc_count).reshape(-1,1)

    def get_wordexcl_features(self, cleaned_text):
        nlp = spacy.load(self.spacy_model)
        matcher = Matcher(nlp.vocab)
        wordexcl_count =[]
        disabled = nlp.select_pipes(disable= ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=-1):
            matcher.add("wordsfollowedbyexclamation",[[{"ORTH": {"REGEX": "([A-Za-z]+\!)"}}]])
            matches = matcher(doc)        
            wordexcl_count.append(len([doc[start:end].text for match_id, start, end in matches]))
            
        return np.array(wordexcl_count).reshape(-1,1)

         
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y=None):
        try:
            if str(type(X)) not in ["<class 'list'>","<class 'numpy.ndarray'>"]:
                raise Exception('Expected list or numpy array got {}'.format(type(X)))

            
            preprocessor1 = cp.SpacyPreprocessor(model = 'en_core_web_sm', lammetize=False, lower = False, 
                                   remove_stop=False )
            preprocessor2 = cp.SpacyPreprocessor(model = 'en_core_web_sm', lammetize=False, lower = False, 
                                   remove_stop=False, remove_punct= False )
            
            feature_names =[]
            if (self.pos_features or self.ner_features or self.punct_features or self.exclam_features or self.free_features or self.tc_features or self.wordexcl_features):
                cleaned_x_count_ner_pos_punct = preprocessor2.fit_transform(X)
            
            if self.count_features:
                cleaned_x_count_features = preprocessor1.fit_transform(X)
                count_features = self.get_count_features(cleaned_x_count_features)
                feature_names.extend(['count_words', 'count_characters',
                                      'count_digits','count_numbers'])
            else:
                count_features = np.empty(shape = (0, 0))
                
            if self.pos_features: 
                pos_features = self.get_pos_features(cleaned_x_count_ner_pos_punct)
                feature_names.extend(['noun_count', 'aux_count', 'verb_count', 'adj_count'])
            else:
                pos_features = np.empty(shape = (0, 0))
                
            if self.ner_features: 
                ner_features =self.get_ner_features(cleaned_x_count_ner_pos_punct)
                feature_names.extend(['ner'])
            else:
                ner_features = np.empty(shape = (0, 0))

            if self.punct_features: 
                punct_features =self.get_punct_features(cleaned_x_count_ner_pos_punct)
                feature_names.extend(['punct_count', 'url_count'])
            else:
                punct_features = np.empty(shape = (0, 0))

            if self.exclam_features: 
                exclam_features =self.get_exclam_features(cleaned_x_count_ner_pos_punct)
                feature_names.extend(['exclam_count'])
            else:
                exclam_features = np.empty(shape = (0, 0)) 

            if self.free_features: 
                free_features =self.get_free_features(cleaned_x_count_ner_pos_punct)
                feature_names.extend(['free_count'])
            else:
                free_features = np.empty(shape = (0, 0)) 

            if self.tc_features: 
                tc_features =self.get_tc_features(cleaned_x_count_ner_pos_punct)
                feature_names.extend(['tc_count'])
            else:
                tc_features = np.empty(shape = (0, 0)) 

            if self.wordexcl_features: 
                wordexcl_features =self.get_wordexcl_features(cleaned_x_count_ner_pos_punct)
                feature_names.extend(['wordexcl_count'])
            else:
                wordexcl_features = np.empty(shape = (0, 0)) 


            return np.hstack((count_features, ner_features, pos_features, punct_features, exclam_features, free_features, tc_features, wordexcl_features)), feature_names
            

        except Exception as error:
            print('An exception occured: ' + repr(error))

