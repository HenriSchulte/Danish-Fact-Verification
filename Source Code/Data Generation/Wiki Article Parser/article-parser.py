from datetime import datetime
import pandas as pd
import re
import urllib.parse
from ftfy import fix_text

n_files = 6
out_df = pd.DataFrame()
start_time = datetime.now()

for n in range(n_files):

    print('Loading file {}...'.format(n))

    source_df = pd.read_json('data/wiki_{:02d}'.format(n), lines=True)
    source_df = source_df[['title', 'text']]

    for index, row in source_df.iterrows():

        print('Step {}/{}'.format(index, len(source_df['title'])), end='\r')

        title = row['title']
        title = fix_text(title).replace(' ', '_')
        linked_entities = []

        # Handle entire article
        rawtext = row['text']    
        abstract = ''
        rawtext = rawtext.splitlines()
        abstract = rawtext[2].strip() if len(rawtext) > 2 else '' # First sentence is at index 2
        abstract = fix_text(abstract)

        if abstract == '':
            continue

        # Handle linked entities in that line
        regex = r"(<a href=\"(.+?)\">(.+?)</a>)"
        links = re.findall(regex, abstract)

        # Store them as linked_entities 
        for link in links:
            linked_entity = link[1]
            linked_entity = urllib.parse.unquote(linked_entity)
            linked_entity = fix_text(linked_entity).strip().replace(' ', '_')
            linked_entities.append(linked_entity)
            # Clean the abstract
            abstract = abstract.replace(link[0], link[2])

        out_df = out_df.append({'Title': title, 'Abstract': abstract, 'Linked Entities': linked_entities}, ignore_index=True)

print('')
out_df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
print('Saved {} articles to file.'.format(len(out_df['Title'])))

print('Parsing took {}'.format(datetime.now()-start_time))