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
    and stores the hash to entity in a dictionary for de-anonymization. A 
    similar process is repeated for categorical columns, without the use
    of NER.
    
    Please use at your own risk, NER identification is not perfect, 
    de-tokenizing is not perfect.
    
    Args
    ----
    df: <pandas.DataFrame>
        dataframe of the data to be anonymized
        
    Methods
    ----
    anonymize:
        Anonymize the dataset with the specified free text / categorical
        columns. Returns an anonymized dataset and a hash dictionary.
    
    de_anonymize:
        De-anonymize the anonymized dataset using the generated anonymized
        data and hash dictionary. This serves only as a quick visual
        test of what de-anonymizing the data will look like.
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
        Returns a tuple of the below items, in order:
        * pandas DataFrame of the anonymized dataframe
        * hash dictionary with key as column name and value as a dictionary of
          hash key to the original data
        """
        # type checking
        assert isinstance(free_text_columns, (list, type(None))),\
            "free_text_columns should be of type list"
        assert isinstance(free_text_additional_regex_to_hash,
                          (dict, type(None))),\
            "free_text_additional_regex_to_hash should be of type dict"
        assert isinstance(categorical_columns, (list, type(None))),\
            "categorical_columns should be of type list"
        assert isinstance(pretrained_model_name, str),\
            "pretrained_model_name should be of type str"
        assert isinstance(label_list, list),\
            "label_list should be of type list"
        assert isinstance(labels_to_anonymize, list),\
            "labels_to_anonymize should be of type list"
        assert set(labels_to_anonymize).issubset(set(label_list)),\
            "elements in labels_to_anonymize should be in labels_list"
        
        self.label_list = label_list
        self.labels_to_anonymize = labels_to_anonymize
        self.pretrained_model_name = pretrained_model_name
        
        df_ = copy.deepcopy(self.df)
        hash_dict = {}
        
        # hash the free text columns
        if free_text_columns is not None:
            
            for col in free_text_columns:
                
                # check for any specified additional regex to hash for the col
                regex = None
                if free_text_additional_regex_to_hash is not None:
                    if col in free_text_additional_regex_to_hash:
                        regex = free_text_additional_regex_to_hash[col]
                
                df_[col], d_ = self._anonymize_free_text(df_[col].tolist(),
                                                         regex)
                hash_dict.update({col: d_})
        
        # hash the categorical columns
        if categorical_columns is not None:
            
            for col in categorical_columns:
                df_[col], d_ = self._anonymize_categorical(df_[col].tolist())
                hash_dict.update({col: d_})
            
        self.anonymized_df = df_
        self.hash_dictionary = hash_dict
        
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
            * Might there be a better way to detokenize tokens?
              - https://github.com/huggingface/transformers/issues/36
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
            sentences = [s for s in row.split(".")]
            anon_sentences = []
            
            for _sentence in sentences:
                
                #-----                
                # NER Prediction & Hashing
                #-----
                
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
                
                #----
                # Regex Hashing
                #----
                # hash regex before NER hashing, but after NER tokenization to
                # prevent (i) regex picking up NER hash values and (ii) NER
                # tokenizing regex hash values
                if additional_regex_to_hash is not None:
                    for regex in additional_regex_to_hash:
                        matches = re.findall(regex, _sentence)
                        
                        if matches:
                            for _m in matches:
                                _hash = (hashlib.md5(str(_m).encode())
                                         .hexdigest())
                                if _hash not in hash_dict_:
                                    hash_dict_.update({_hash: _m})
                                _sentence = _sentence.replace(_m, _hash)
                
                #----
                # End Regex Hashing, Continue NER Prediction & Hashing
                #----
                
                words_to_anonymize = []
                prev_i = -1
                prev_label = ""
                
                # loop over every token prediction
                for i, pred in enumerate(_preds):
                    # get prediction label
                    label = label_list[pred]
                    
                    # check if label should be anonymized
                    if label in labels_to_anonymize:
                        # get original word
                        word = tokens[i] # detokens[i]
                        
                        # if consecutive tokens with same label, concat words
                        # else append word
                        if ((i-1 == prev_i) & (label == prev_label)):
                            words_to_anonymize[-1] = (
                                words_to_anonymize[-1] + " " + word 
                            )
                        else:
                            words_to_anonymize.append(word)
                            
                        prev_i = i
                        prev_label = label
                                
                for _word in words_to_anonymize:
                    
                    # attempt to "detokenize" word here by using
                    # the tokens as a regex
                    
                    # remove ## & whitespaces and do a regex match with any 
                    # \s\W in between any characters
                    _regex = [c.lower() for c in re.sub("##| ", "", _word)]
                    word_regex = "[\s\W]{0,1}".join(_regex)
                    
                    match = re.search(word_regex, _sentence,
                                      flags=re.IGNORECASE)
                    
                    # there should only be one match
                    if match:
                        word = match[0]
                           
                        # hash the word
                        _hash = hashlib.md5(str(word).encode()).hexdigest()
                        
                        # update hash dictionary
                        if _hash not in hash_dict_:
                            hash_dict_.update({_hash: word})
                        
                        # split by hashes to prevent parts of the hash from
                        # being replaced
                        _sen = re.split("("+ "|".join(hash_dict_.keys()) + ")",
                                        _sentence)
                        
                        # replace non hash words
                        _s = [s.replace(word, _hash) if s 
                              not in hash_dict_.keys() else s for s in _sen]
                        
                        # recreate sentence
                        _sentence = "".join(_s)
                
                anon_sentences.append(_sentence)
                
            anon_sentence = ".".join(anon_sentences)
            anonymized_data.append(anon_sentence)
            
        return anonymized_data, hash_dict_

    
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
            if _hash not in hash_dict_:
                hash_dict_.update({_hash: cat})
            
        return anonymized_data, hash_dict_

        
    def de_anonymize(self):
        """ 
        Returns
        ----
        pandas DataFrame containing the de-anonymized data
        """
        return de_anonymize_data(self.anonymized_df, self.hash_dictionary)


def de_anonymize_data(anonymized_df, hash_dictionary):
    """De-anonymize a dataframe given a corresponding hash dictionary
    
    Args
    ----
    anonymized_df: <pandas.DataFrame>
        dataframe of the anonymized data

    hash_dictionary: <dict>
        dictionary consisting of key as column name and value as a 
        dictionary of hash key to the original data

    Returns
    ----
    pandas DataFrame containing the de-anonymized data
    """
    # type checking
    assert isinstance(anonymized_df, (pd.DataFrame, type(None))),\
        "anonymized_df should be a pandas DataFrame"
    assert isinstance(hash_dictionary, (dict, type(None))),\
        "hash_dictionary should be of type dict"

    df_ = copy.deepcopy(anonymized_df)

    # de_anonymize free text columns
    for col, _hash_dict in hash_dictionary.items():
        for _hash, _original in _hash_dict.items():
            df_[col] = df_[col].str.replace(_hash, _original)

    return df_