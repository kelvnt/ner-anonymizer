# NER Anonymizer
[![PyPI version](https://badge.fury.io/py/ner-anonymizer.svg)](https://badge.fury.io/py/ner-anonymizer)

This package contains some developmental tools to anonymize a pandas dataframe.

NER Anonymizer contains a class `DataAnonymizer` which handles anonymization for both free text and categorical columns in a pandas dataframe:
* For free text columns, it uses a pretrained model from the [transformers](https://huggingface.co/transformers/) package to perform named entity recognition (NER) to pick up user specified entities such as location and person, generate a MD5 hash for the entity, replaces the entity with the hash, and stores the hash to entity in a dictionary
* For categorical columns, it simply generates a MD5 hash for every category, replaces the category with the hash, and stores the hash to category in a dictionary

The saved dictionary can then be used for de-anonymization and the original dataset is obtained. Referential integrity is preserved as the same hash will be generated for the same category / entity.

## Installation
Install the package with pip

    pip install ner-anonymizer

## Example Usage
The package uses the NER model [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) by default. To anonymize a particular pandas dataframe, `df`, using a pretrained NER model:

    import ner_anonymizer

    # to anonymize
    anonymizer = ner_anonymizer.DataAnoynmizer(df)
    anonymized_df, hash_dictionary = anonymizer.anonymize(
        free_text_columns=["free_text_column_1", "free_text_column_2"],
        categorical_columns=["categorical_column_1"],
        pretrained_model_name="dslim/bert-base-NER",
        label_list=["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
        labels_to_anonymize=["B-PER", "I-PER", "B-LOC", "I-LOC"]
    )

    # to de-anonymize
    de_anonymized_df = ner_anonymizer.de_anonymize_data(df, hash_dictionary)

You may specify for the argument `pretrained_model_name` any available pre-trained NER model from the [transformers](https://huggingface.co/transformers/) package in the links below (do note that you will need to specify the labels that the NER model uses, `label_list`, and from that list, the labels you want to anonymize, `labels_to_anonymize`):
* https://huggingface.co/transformers/pretrained_models.html
* https://huggingface.co/models

You may also view an example notebook in the following directory `examples/example_usage.ipynb`.
