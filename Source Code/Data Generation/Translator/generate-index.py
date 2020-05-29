import pandas as pd
import json

ids =  []
file_names = []
for i in range(1, 110):
    wiki_file_name = 'wiki-{:03d}.jsonl'.format(i)
    print('Indexing file {}'.format(wiki_file_name), end='\r')
    wiki_file_path = 'data/wiki-pages/' + wiki_file_name
    with open(wiki_file_path) as wiki_json_file:
        for line in wiki_json_file:
            line_decoded = json.loads(line)
            ids.append(line_decoded['id'])
            file_names.append(wiki_file_name)

df = pd.DataFrame()
df['id'] = ids
df['file_name'] = file_names
df.to_json('wiki-index.jsonl', orient='records', lines=True)            