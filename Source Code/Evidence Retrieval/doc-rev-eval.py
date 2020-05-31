import pandas as pd
import nltk
import ftfy

def fix_text(series):
    return series.apply(lambda x: ftfy.fix_text(x))

def split_evidence(series, tokenizer):
    series = series.apply(lambda x: x.replace('"', '')) # Replacing off quotation marks
    return series.apply(lambda x: tokenizer.tokenize(x))

def compute_intersection(series):
    return [sent for sent in series.evidence_gold if sent in series.evidence_retr]

def compute_precision(series):
    return len(series.evidence_both) / float(len(series.evidence_retr)) if len(series.evidence_retr) else 0

def compute_recall(series):
    return len(series.evidence_both) / float(len(series.evidence_gold))

gold_path = "../Classifier/data/test_data.jsonl"
retrieved_path = "data/retrieval_9_9.jsonl"

gold_df = pd.read_json(gold_path, lines=True)
retrieved_df = pd.read_json(retrieved_path, lines=True)

# Fix encoding
retrieved_df.evidence = fix_text(retrieved_df.evidence)
retrieved_df.claim = fix_text(retrieved_df.claim)

# Split evidence to sentences
tokenizer = nltk.data.load('tokenizers/punkt/danish.pickle')
gold_df.evidence = split_evidence(gold_df.evidence, tokenizer)
retrieved_df.evidence = split_evidence(retrieved_df.evidence, tokenizer)

df = gold_df.merge(right=retrieved_df.evidence, on=gold_df.index, suffixes=('_gold', '_retr')).drop('key_0', axis=1)
df = df.query('label != "NotEnoughInfo"')

df['evidence_both'] = df.apply(lambda x: compute_intersection(x), axis=1)
df['precision'] = df.apply(lambda x: compute_precision(x), axis=1)
df['recall'] = df.apply(lambda x: compute_recall(x), axis=1)
df.to_json('results.jsonl', orient='records', lines=True)

precision = df.precision.mean()
recall = df.recall.mean()
f1 = 2 * ((precision * recall)/(precision + recall))
print('Mean Precision: {}'.format(precision))
print('Mean Recall: {}'.format(recall))
print('F1: {}'.format(f1))