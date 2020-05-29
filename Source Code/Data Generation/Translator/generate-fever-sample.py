import os, requests, uuid, json
from datetime import datetime
import pandas as pd
import time

print('Reading training data...')
data = []
with open('data/train.jsonl') as train_json_file:
    for line in train_json_file:
        line_decoded = json.loads(line)
        data.append(line_decoded)

# Read index
print('Reading index...')
index_df = pd.read_json('wiki-index.jsonl', lines=True)

# Convert to dataframe
train_df = pd.DataFrame(data)
train_df = train_df.sample(1000)

# Loop over claims and translate each
print('Processing claims...')
claims_translated = []
evidences_translated = []
evidences_en = []

i = 1
for index, row in train_df.iterrows():
    print('Processing claim {}/{}...'.format(i, len(train_df['id']+1)), end='\r')
    i += 1
    # Claim
    claim_en = row['claim']

    # Lookup evidence
    claim_evidence_list = row['evidence'][0]
    claim_evidence_translated = []
    claim_evidence_en = []

    for claim_evidence in claim_evidence_list:
        evidence_id = claim_evidence[2]
        if evidence_id is None: evidence_en = ''
        
        index_row = index_df.query('id == \"{}\"'.format(evidence_id))
        if len(index_row['file_name']) == 0: # If evidence id can't be found in index
            evidence_da = ''
            evidence_en = ''
        
        else:
            file_name = index_row['file_name'].values[0]

            # Look up evidence in file
            wiki_file = pd.read_json('data/wiki-pages/' + file_name, lines=True)
            evidence_row = wiki_file.query('id == \"{}\"'.format(evidence_id))
            evidence_en = evidence_row['text'].values[0]

        claim_evidence_en.append(evidence_en)
    evidences_en.append(claim_evidence_en)

min_len = len(evidences_en)
train_df = train_df.head(min_len)
train_df.insert(4, 'evidence_en', evidences_en[:min_len+1])
print('Saving claims and evidence...')
# Save dataframe as json
train_df_en = train_df[['id', 'label', 'claim', 'evidence_en']]
train_df_en.to_json('out/{}_en.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
