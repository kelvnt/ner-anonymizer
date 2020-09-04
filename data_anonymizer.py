import numpy as np
import pandas as pd
import torch
import hashlib
from transformers import AutoTokenizer, AutoModelForTokenClassification
import copy
import re


class DataAnonymizer:
    """Anonymizes dataset and provides a hash dictionary of the operation
    
    DataAnonymizer handles anonymization in free text columns by using
    named entity recognition (NER) with a pretrained mdoel from the
    transformers package to pick up entities such as location and person, 
    generate a MD5 hash for the entity, replaces the entity with the hash,
    and stores the hash to entity in a dictionary for de-anonymization. The 
    same process is done for categorical columns, without the use of
    NER. 
    
    Please use at your own risk, NER identification is not perfect.
    
    Args
    ----
    df: <pandas.DataFrame>
        dataframe of the data to be anonymized
    
    """
    def __init__(self, df):
        # type checking
        assert isinstance(df, pd.DataFrame), "df should be a pandas DataFrame"
        
        self.df = copy.deepcopy(df)
        
    def anonymize(self,
                  free_text_columns=None,
                  free_text_additional_regex_to_hash=None,
                  categorical_columns=None,
                  pretrained_model_name="dslim/bert-base-NER",
                  label_list=["O", "B-MISC", "I-MISC", "B-PER", "I-PER",
                              "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
                  labels_to_anonymize=["B-PER", "I-PER", "B-LOC", "I-LOC"]
                 ):
        """
        Args
        ----
        free_text_columns: <list of str>
            list of column headers which contain free text columns to
            be anonymized
            
        free_text_additional_regex_to_hash: <dict>
            dictionary with key as a column name in `free_text_columns` and
            value as a list of regex patterns to search for in the
            specified free text column to hash
        
        categorical_columns: <list of str>
            list of column headers which contain categorical columns to
            be anonymized
            
        pretrained_model_name: <str>
            name of pretrained model from the transformers package to use
            for NER. Available model list below:
            * https://huggingface.co/transformers/pretrained_models.html
            * https://huggingface.co/models
            
        label_list: <list of str>
            label list used in the defined `pretrained_model_name`
            
        labels_to_anonymize: <list of str>
            list of entities in the defined `label_list` to be anonymized
            
        Returns
        ----
        df_: <pandas.DataFrame>
            anonymized dataframe
        
        hash_dict: <dictionary>
            dictionary with key as column name and value as a dictionary of
            hash key to the original data
        """
        # arguments checking
        assert isinstance(free_text_columns, (list, type(None))), "free_text_columns should be of type list"
        assert isinstance(free_text_additional_regex_to_hash, (dict, type(None))), "free_text_additional_regex_to_hash should be of type dict"
        assert isinstance(categorical_columns, (list, type(None))), "categorical_columns should be of type list"
        assert isinstance(pretrained_model_name, str), "pretrained_model_name should be of type str"
        assert isinstance(label_list, list), "label_list should be of type list"
        assert isinstance(labels_to_anonymize, list), "labels_to_anonymize should be of type list"
        assert set(labels_to_anonymize).issubset(set(label_list)), "elements in labels_to_anonymize should be in labels_list"
        
        self.label_list = label_list
        self.labels_to_anonymize = labels_to_anonymize
        self.pretrained_model_name = pretrained_model_name
        
        df_ = copy.deepcopy(self.df)
        hash_dict = {}
        
        if free_text_columns is not None:
            _free_text_dict = {}
            for col in free_text_columns:
                
                # check if there is any additional regex to hash for the col
                regex = None
                if free_text_additional_regex_to_hash is not None:
                    if col in free_text_additional_regex_to_hash:
                        regex = free_text_additional_regex_to_hash[col]
                
                df_[col], d_ = self._anonymize_free_text(df_[col].tolist(),
                                                         regex)
                _free_text_dict.update({col: d_})
            hash_dict.update({"free_text": _free_text_dict})
                
        if categorical_columns is not None:
            _categorical_dict = {}
            for col in categorical_columns:
                df_[col], d_ = self._anonymize_categorical(df_[col].tolist())
                _categorical_dict.update({col: d_})
            hash_dict.update({"categorical": _categorical_dict})
        
        return df_, hash_dict
    
                
    def _anonymize_free_text(self, l, additional_regex_to_hash):
        """
        Args
        ----
        l: <list>
            list of free text paragraphs
            
        additional_regex_to_hash: <list>
            list of regex patterns to hash
            
        Notes:
            * might need to clean special characters
            * how to ensure character separation is always a space
            * should save original word, not the token
            * how to de-tokenize berttokenizer?
        """
        label_list = self.label_list
        labels_to_anonymize = self.labels_to_anonymize
        
        # load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name
        )
        model = AutoModelForTokenClassification.from_pretrained(
            self.pretrained_model_name
        )
        
        anonymized_data = []
        hash_dict_ = {}
        
        for row in l:
            # each s should not be longer than 512 characters
            sentences = [s[:512] for s in row.split(".")]
            anon_sentences = []
            
            for _sentence in sentences:
                
                # handle the regex portion
                if additional_regex_to_hash is not None:
                    for regex in additional_regex_to_hash:
                        matches = re.findall(regex, _sentence)
                        
                        if matches:
                            for _m in matches:
                                _hash = hashlib.md5(str(_m).encode()).hexdigest()
                                if _hash not in hash_dict_:
                                    hash_dict_.update({_hash: _m})
                                _sentence = _sentence.replace(_m, _hash)
                                
                # handle transformers NER recognition
                
                # tokenize sentence
                tokens = tokenizer.tokenize(
                    tokenizer.decode(tokenizer.encode(_sentence))
                )
                inputs = tokenizer.encode(
                    _sentence, return_tensors="pt"
                )
                
                # predict entity
                outputs = model(inputs)[0]
                _preds = torch.argmax(outputs, dim=2)[0].tolist()
                
                # detokenize tokens
                # detokens = self._detokenize(tokens)
                
                # get words of entities to anonymize
                words_to_anonymize = []
                
                for i, pred in enumerate(_preds):
                    label = label_list[pred]
                    
                    if label in labels_to_anonymize:
                        word = tokens[i].replace("##", "") # detokens[i]
                        
                        if len(words_to_anonymize) == 0:
                            words_to_anonymize.append(word)
                            prev_i = i
                            prev_label = label
                            
                        elif len(words_to_anonymize) > 0:
                            if (i-1 == prev_i) & (label == 
                                                  prev_label):
                                words_to_anonymize[-1] = (
                                    words_to_anonymize[-1] + " " + word 
                                )
                                prev_i = i
                                prev_label = label
                                
                            else:
                                words_to_anonymize.append(word)
                                prev_i = i
                                prev_label = label
                
                for _word in words_to_anonymize:
                    _hash = hashlib.md5(str(_word).encode()).hexdigest()
                    if _hash not in hash_dict_:
                        hash_dict_.update({_hash: _word})
                        
                    _sentence = _sentence.replace(_word, _hash)
                
                anon_sentences.append(_sentence)
                
            anon_sentence = ". ".join(anon_sentences)
            anonymized_data.append(anon_sentence)
            
        return anonymized_data, hash_dict_

    
    def _detokenize(self, tokens):
        """
        to-do: lengths does not match..
        """
        
        is_subtoken = lambda x: True if x[:2] == "##" else False

        restored_text = []
        for i in range(len(tokens)):
            if not is_subtoken(tokens[i]) and (i+1)<len(tokens) and is_subtoken(tokens[i+1]):
                restored_text.append(tokens[i] + tokens[i+1][2:])
                if (i+2)<len(tokens) and is_subtoken(tokens[i+2]):
                    restored_text[-1] = restored_text[-1] + tokens[i+2][2:]
            elif not is_subtoken(tokens[i]):
                restored_text.append(tokens[i])
                
        return restored_text

    def _anonymize_categorical(self, l):
        """
        Args
        ----
        l: <list>
            list of categorical data
        """
        anonymized_data = []
        hash_dict_ = {}
        
        for cat in l:
            _hash = hashlib.md5(str(cat).encode()).hexdigest()
            anonymized_data.append(_hash)
            hash_dict_.update({_hash: cat})
            
        return anonymized_data, hash_dict_

        
    def de_anonymize(self, anonymized_df, hash_dictionary):
        """
        Args
        ----
        anonymized_df: <pandas.DataFrame>
            dataframe of the anonymized data
        
        hash_dictionary: <dict>
            dictionary consisting of key as column name and value as a 
            dictionary of hash key to the original data
            
        Returns
        ----
        df_: <pandas.DataFrame>
            data frame containing the de-anonymized data
        """
        # type checking
        assert isinstance(hash_dictionary, dict), "hash_dictionary should be of type dict"
        
        df_ = copy.deepcopy(anonymized_df)
        
        if "free_text" in hash_dictionary:
            for col, _hash_dict in hash_dictionary["free_text"].items():
                for _hash, _original in _hash_dict.items():
                    df_[col] = df_[col].str.replace(_hash, _original)
                    
        if "categorical" in hash_dictionary:
            for col, _hash_dict in hash_dictionary["categorical"].items():
                df_[col] = df_[col].replace(_hash_dict)
                
        return df_