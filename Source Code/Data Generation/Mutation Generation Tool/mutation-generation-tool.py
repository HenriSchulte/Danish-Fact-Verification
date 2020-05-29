from datetime import datetime
import pandas as pd
import random

# source_df = pd.read_csv('data/dawiki-latest-abstract.csv')
source_df = pd.read_json('data/data.jsonl', lines=True)
source_df = source_df[['claim', 'entity', 'evidence', 'linked entities']]
print('{} rows loaded.\n'.format(len(source_df['claim'])))

mutations_df = pd.DataFrame()
counter = 1

for index, source_row in source_df.iterrows():
    print('\n------- MUTATION OF CLAIM NUMBER {}/{} -------\n'.format(counter, len(source_df['claim'])))
    counter += 1

    source_claim = source_row['claim']
    source_entity = source_row['entity']
    source_evidence = source_row['evidence']
    source_linked_entities = source_row['linked entities']

    # Save original claim 
    mutations_df = mutations_df.append({'claim': source_claim, 'entity': source_entity, 'evidence': source_evidence, 'linked entities': source_linked_entities}, ignore_index=True)
     
    print('Source entity:')
    print(source_entity)
    print('Source claim:')
    print(source_claim)
    print('')

    quit_flag = False

    while True:
        user_input = input("Write mutation or hit enter to skip  or 'quit'\n")

        if user_input == 'quit':
            quit_flag = True
            break
        elif user_input == '':
            print('')
            break
        else:
            user_input = user_input.strip()    
            mutations_df = mutations_df.append({'claim': user_input, 'entity': source_entity, 'evidence': source_evidence, 'linked entities': source_linked_entities}, ignore_index=True)
        print('')

    if quit_flag:
        break


print('No more claims. Good job, Old Sport!')

if not mutations_df.empty:
    mutations_df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
    print('Saved {} mutations to file.'.format(len(mutations_df['claim'])))
else:
     print('No mutations were saved.')