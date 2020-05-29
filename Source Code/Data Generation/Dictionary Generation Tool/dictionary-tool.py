import pandas as pd
from datetime import datetime

print('Reading files...')
wiki_articles = pd.read_json('../CommonData/dawiki-latest-pages-articles-parsed.jsonl', lines=True)
mutations = pd.read_json('data/data.jsonl', lines=True)

df_dictionary = pd.DataFrame()

def lookup(entity, value):
    # lower case the search to avoid silly bugs 
    row = wiki_articles.loc[wiki_articles['Title'].str.lower() == entity.lower()]
    return row[value].values[0] if row[value].values.size else ''

print('Making dictionary...')
counter = 1
entity = ''
# Lookup mutation entity in articles
for index, row in mutations.iterrows():
    print('Step {}/{}'.format(counter, len(mutations)), end='\r')
    if entity !=  row['entity']:
        entity = row['entity']
        linked_entities = row['linked entities']
        dictionary = []

        # For every linked entity, look up in articles to find the abstract 
        for link in linked_entities: 
            abstract = lookup(link, 'Abstract')

            # Save abstracts with mutation 
            if abstract != '':
                dictionary_entry = {'entity': link, 'abstract': abstract}
                dictionary.append(dictionary_entry)
            
        # Final structure {'entity': ..., 'claim:' ..., 'evidence': ..., [{'entity1': ...}, {'entity2': ...}, ... {'entityN': ...}]}
    df_dictionary = df_dictionary.append({'claim': row['claim'], 'entity': entity, 'evidence': row['evidence'], 'dictionary': dictionary }, ignore_index=True)
    counter += 1

print('')
df_dictionary.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
print('Saved {} claims + dictionaries to file.'.format(len(df_dictionary['entity'])))





