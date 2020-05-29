from datetime import datetime
import pandas as pd
import random

source_df = pd.read_json('../CommonData/dawiki-latest-pages-articles-parsed.jsonl', lines=True)
source_df = source_df[['Title', 'Abstract', 'Linked Entities']]
print('{} rows loaded.'.format(len(source_df['Title'])))

claims_df = pd.DataFrame()
counter = 1

while True:
    source_row = source_df.sample(1) # Sample one row
    source_title = source_row['Title'].values[0]
    source_abstract = source_row['Abstract'].values[0]
    source_linked_entities = source_row['Linked Entities'].values[0]
    
    source_sentence = source_abstract

    print('\n------- CLAIM NUMBER {} -------\n'.format(counter))

    print('Source entity:')
    print(source_title)
    print('Source sentence:')
    print(source_abstract)
    print('')

    user_input = input("Enter claim, hit enter to skip or write 'quit'\n")

    if user_input == 'quit':
        break
    elif user_input != '': # Add some extra validation here
        counter +=1
        claims_df = claims_df.append({'claim': user_input, 'evidence': source_sentence, 'entity': source_title, 'linked entities': source_linked_entities}, ignore_index=True)

if not claims_df.empty:
    claims_df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
    print('Saved {} claims to file.'.format(len(claims_df['claim'])))
else:
    print('No claims were saved.')