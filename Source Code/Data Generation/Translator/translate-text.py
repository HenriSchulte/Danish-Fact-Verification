# Code from: https://docs.microsoft.com/en-us/azure/cognitive-services/translator/quickstart-translate?pivots=programming-language-python

# -*- coding: utf-8 -*-
import os, requests, uuid, json
from datetime import datetime
import pandas as pd
import time

def translate(text, delay_factor=1):
    """Translate given text to Danish. Return translated text."""
    print('Translating {}...'.format(text[:50]))
    if (len(text) > 5000): # Text may not exceed 500 characters
        print('Request exceeds character limit. Skipping...')
        return 'N/A'

    time.sleep(len(text) / 500 * delay_factor) # Artifical delay to not overwhelm the server

     # Set request body for request
    body = [{
        'text': text
    }]

    # Make the POST request
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    # Validate
    try:
        translation = response[0]['translations'][0]['text'] # navigating through json object response
        return translation
    except:
        print('Translation failed. Skipping...')
        return 'N/A'

# Extract key and endpoint from environment variables
print('Reading environment variables...')
key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
if not key_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(key_var_name))
subscription_key = os.environ[key_var_name]

endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
if not endpoint_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(endpoint_var_name))
endpoint = os.environ[endpoint_var_name]

# Set up parameters
print('Setting translation parameters...')
path = '/translate?api-version=3.0'
params = '&from=en&to=da'
constructed_url = endpoint + path + params

# Pass subscription key for authentication
headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

# Read dataset
print('Reading training data...')
data = []
with open('data/train.jsonl') as train_json_file:
    for line in train_json_file:
        line_decoded = json.loads(line)
        data.append(line_decoded)

# Read rosetta stone
print('Reading Rosetta Stone...')
rosetta_df = pd.read_json('rosetta-stone.jsonl', lines=True)

# Read index
print('Reading index...')
index_df = pd.read_json('wiki-index.jsonl', lines=True)
    
# Convert to dataframe
train_df = pd.DataFrame(data)

# Sample
train_df = train_df.sample(300)

# Loop over claims and translate each
print('Processing claims...')
claims_translated = []
evidences_translated = []
evidences_en = []

try:
    i = 1
    for index, row in train_df.iterrows():
        print('Processing claim {}/{}...'.format(i, len(train_df['id']+1)))
        i += 1
        # Claim
        claim_en = row['claim']

        claim_da = translate(claim_en)
        claims_translated.append(claim_da)

        # Evidence

        # Lookup evidence
        claim_evidence_list = row['evidence'][0]
        claim_evidence_translated = []
        claim_evidence_en = []

        for claim_evidence in claim_evidence_list:
            evidence_id = claim_evidence[2]
            if evidence_id is None:
                evidence_da = ''
                evidence_en = ''
            else:
                # Determine whether we already translated it before
                translation_row = rosetta_df.query('id == \"{}\"'.format(evidence_id))
                if len(translation_row['id']):
                    print('Thank you, Napoleon!')
                    evidence_da = translation_row['evidence_da'].values[0]
                    evidence_en = translation_row['evidence_en'].values[0]
                else:

                    # Look up file name in index
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

                        # Translate evidence
                        evidence_da = translate(evidence_en)

                        rosetta_df = rosetta_df.append({'id': evidence_id, 'evidence_da': evidence_da, 'evidence_en': evidence_en}, ignore_index=True)

            claim_evidence_translated.append(evidence_da)
            claim_evidence_en.append(evidence_en)
        evidences_translated.append(claim_evidence_translated)
        evidences_en.append(claim_evidence_en)

except Exception as e:
    print('Exception: ', e)

# Recovery after exception
min_len = min(len(claims_translated), len(evidences_translated), len(evidences_en))
train_df = train_df.head(min_len)

# Save translations to dataframe
train_df.insert(4, 'claim_da', claims_translated[:min_len+1])
train_df.insert(5, 'evidence_da', evidences_translated[:min_len+1])
train_df.insert(6, 'evidence_en', evidences_en[:min_len+1])

print('Saving claims and evidence...')
# Save dataframe as json
train_df_en = train_df[['id', 'label', 'claim', 'evidence_en']]
train_df_en.to_json('out/{}_en.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
train_df_da = train_df[['id', 'label', 'claim_da', 'evidence_da']]
train_df_da.to_json('out/{}_da.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)

print('Saving rosetta stone...')
# Save rosetta stone
rosetta_df.to_json('rosetta-stone.jsonl', orient='records', lines=True)

print("SUCCESS!")