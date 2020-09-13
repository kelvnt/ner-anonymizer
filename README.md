# NER Anonymizer
This repository contains some developmental tools to anonymize a pandas dataframe.

NER Anonymizer contains a class `DataAnonymizer` which handles anonymization in free text columns by using named entity recognition (NER) with a pretrained model from the [transformers](https://huggingface.co/transformers/) package to pick up entities such as location and person, generate a MD5 hash for the entity, replaces the entity with the hash, and stores the hash to entity in a dictionary for de-anonymization. A similar process is repeated for categorical columns, without the use of NER.

## Example Usage
Open a terminal and run the following lines (this assumes you have python 3 installed):

    git clone https://github.com/kelvnt/data_anonymizer.git
    cd data_anonymizer
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    jupyter-lab

Open `example_usage.ipynb` to explore how DataAnonymizer works.
