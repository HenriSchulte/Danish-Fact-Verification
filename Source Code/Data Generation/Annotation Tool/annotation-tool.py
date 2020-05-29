from datetime import datetime
import pandas as pd
import random, re

source_df = pd.read_json('data/data.jsonl', lines=True)
source_df = source_df[['claim', 'entity', 'evidence', 'dictionary']]
print('{} rows loaded in claims.'.format(len(source_df['claim'])))

annotation_df = pd.DataFrame()
source_df = source_df.sample(frac=1) # Shuffling the rows 
counter = 1

for index, source_row in source_df.iterrows():
    print('')

    source_claim = source_row['claim']
    source_entity = source_row['entity']
    source_evidence = source_row['evidence']
    source_dict = source_row['dictionary']

    print('------- CLAIM {}/{} -------\n'.format(counter, len(source_df['claim'])))
    counter += 1

    print('Source entity:')
    print(source_entity)
    print('Source claim:')
    print(source_claim)
    print('Source evidence:')
    print(source_evidence)
    print('')

    print('------- DICTIONARY -------\n')

    for index, pair in enumerate(source_dict):
        abstract = pair['abstract']
        print('[{}] {}\n'.format(index, abstract))

    print('-------- ANNOTATE --------\n')

    invalid_input = True
    quit_flag = False
    continue_flag = False

    while invalid_input:      # Keep asking for input until valid
        invalid_input = False

        user_input = input("[S] Supported \n[R] Refuted \n[N] Not Enough Info \nProvide useful sentences by their index \nEnter to skip or 'quit'\n")

        if user_input == 'quit': 
            quit_flag = True
            break

        elif user_input == '':
            continue_flag = True
            break 

        annotation = ''
        used_sentences = [source_evidence]
        used_entities = [source_entity]

        input_tokens = re.split(r"\s+", user_input.strip())
        annotation_token = input_tokens[0]

        if annotation_token.lower() == 's':
            annotation = 'Supported'

        elif annotation_token.lower() == 'r': 
            annotation = 'Refuted'

        elif annotation_token.lower() == 'n':
            annotation = 'NotEnoughInfo'
            used_sentences[0] = ''
        else: 
            invalid_input = True
            print('Invalid annotation input detected. Try again.')
            continue

        try:
            for token in input_tokens[1:]:
                token = int(token)
                dict_object = source_dict[token]

                used_entities.append(dict_object['entity'])
                used_sentences.append(dict_object['abstract'])
        except:
            invalid_input = True
            print('Invalid sentence input detected. Try again.')

    if quit_flag: 
        break
    elif continue_flag: 
        continue
    annotation_df = annotation_df.append({'claim': source_claim, 'entity': used_entities, 'evidence': used_sentences, 'label': annotation}, ignore_index=True)

else:
    print('No more claim + evidence pairs to annotate. Good job, Old Sport!')

if not annotation_df.empty:
    annotation_df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
    print('Saved {} annotations to file.'.format(len(annotation_df['claim'])))

else:
      print('No annotations were saved.')



