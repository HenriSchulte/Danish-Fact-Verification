import json
import os
import torch
import transformers
import math
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import glob
import ftfy
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample

#--------
# Method Definitions
#--------

def read_config():
    """ Take the config path as a string and return the config parameters in a map. """
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)
    return (config['train_data_path'], config['validation_data_path'], config['test_data_path'],
             config['pretrained_model'], config['max_len'], config['batch_size'], config['num_labels'],
             config['learning_rate'], config['num_epochs'], bool(config['enable_plotting']), 
             config['output_dir'], bool(config['enable_training']), config['data_sample'],
             bool(config['enable_cuda']), bool(config['enable_test']), config['model_save_dir'], bool(config['enable_test_export']))

def check_required_paths():
    """ Check that output path exists, otherwise create it. """
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print('Output directory was not found. Created /{}.'.format(OUTPUT_DIR))
    if ENABLE_TRAINING:
        if not os.path.isdir(MODEL_SAVE_DIR):
            os.mkdir(MODEL_SAVE_DIR)
            print('Model directory was not found. Created /{}.'.format(MODEL_SAVE_DIR))

def drop_duplicate_claims(df):
    """ Drops rows with duplicate values in claim column. Modifies DF in place! """
    len_with_dupes = len(df['claim'])
    df.drop_duplicates(subset='claim', inplace=True)
    len_no_dupes = len(df['claim'])
    print('Dropped {} duplicate rows.'.format(len_with_dupes - len_no_dupes))
    return df

def concatenate_evidence(df):
    """ Concatenate the evidence for each claim into one string. Edits df in place. """
    df.evidence = df.evidence.transform(lambda x: ' '.join(x))
    return df

def convert_labels(df, columns=['label'], reverse=False):
    """ Convert labels in provided columns to numeric values (or back to text). Edits the dataframe in place. """
    if reverse:
        labels_vals = {
            0: 'Refuted',
            1: 'Supported',
            2: 'NotEnoughInfo'
        }
    else:
        labels_vals = {
            'Refuted': 0,
            'Supported': 1, 
            'NotEnoughInfo': 2    
        }

    for col in columns:
        df[col] = df[col].apply(lambda x: labels_vals[x])
    return df

def balance_data(df):
    """
    Balance dataset by oversampling minority classes to size of majority class.
    Calculate class weights by giving minority classes proportionally higher weights.
    Return shuffled dataframe and class weights criterion.
    """
    supported_df = df[df['label'] == 1]
    refuted_df = df[df['label'] == 0]
    nei_df = df[df['label'] == 2]

    major_len = max([len(supported_df.label), len(refuted_df.label), len(nei_df.label)])
    combined_df = pd.DataFrame(columns=['claim', 'entity', 'evidence', 'label'])
    class_weights = []

    for df in [supported_df, refuted_df, nei_df]:
        weight = major_len / float(len(df.label))      # Calculate class weight based on proportional size of class: smaller size -> larger weight
        class_weights.append(weight)
        df = resample(df, replace=True, n_samples=major_len)    # Oversample
        combined_df = combined_df.append(df)

    shuffled_df = combined_df.sample(frac=1)   # Shuffle
    class_weights = torch.FloatTensor(class_weights).cuda() if ENABLE_CUDA else torch.FloatTensor(class_weights)        # Move to GPU
    criterion = CrossEntropyLoss(weight=class_weights)
    return shuffled_df, criterion

def tokenize_inputs(df):
    """ Return tokenized input ids and token type ids. """
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL, do_lower_case=True)
    input_ids_0 = generate_input_ids(df.claim, tokenizer)
    input_ids_1 = generate_input_ids(df.evidence, tokenizer)
    token_type_ids = transform_input_ids(input_ids_0, input_ids_1, tokenizer.create_token_type_ids_from_sequences)
    input_ids = transform_input_ids(input_ids_0, input_ids_1, tokenizer.build_inputs_with_special_tokens) # Add special tokens like [CLS] and [SEP]
    if not os.path.exists(MODEL_SAVE_DIR + 'vocab.txt'): 
        tokenizer.save_vocabulary(MODEL_SAVE_DIR) # Save vocab if not already in model directory
    return input_ids, token_type_ids

def generate_input_ids(sequences, tokenizer):
    """ Take a list of sequences (claim or evidence) and a tokenizer. Return input ids for sequences as a list."""
    tokenized_text = [tokenizer.tokenize(seq) for seq in sequences] # Borgmester -> Borg - mester
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_text] # Borg -> float
    return input_ids

def transform_input_ids(input_ids_0, input_ids_1, tokenizer_func):
    """
    Take the input ids for sequences 0 and 1 (claim and evidence) and a tokenizer function.
    Apply function to tuples of claim-evidence.
    Return list of token type ids.
    """
    transformed_ids = list(map(
        lambda ids_tuple: tokenizer_func(ids_tuple[0], ids_tuple[1]),
        zip(input_ids_0, input_ids_1)
    ))
    return transformed_ids

def generate_attention_masks(input_ids):
    """ 
    For every sequence in the input_id, generate attention masks.
    1 for every non-zero id, 0 for padding.
    Return attention masks as list.
    """
    attention_masks = list(map(lambda x: [float(i > 0) for i in x], input_ids))
    return attention_masks

def initialise_dataloader(input_ids, attention_masks, labels, token_type_ids):
    """
    Taking parameters as iterables and return a dataloader.
    """

    input_tensor = create_padded_tensor(input_ids)
    mask_tensor = create_padded_tensor(attention_masks)
    type_tensor = create_padded_tensor(token_type_ids)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    if ENABLE_CUDA: label_tensor = label_tensor.cuda()

    dataset = TensorDataset(input_tensor, mask_tensor, label_tensor, type_tensor)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

    return dataloader

def initialise_model():
    """ Initialise model. Move to GPU if CUDA is enabled. Return model. """
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=NUM_LABELS)
    if ENABLE_CUDA: model = model.cuda()
    return model

def create_padded_tensor(sequence):
    """
    Take a list of variable length lists.
    Truncate and pad them all to MAX_LEN.
    Return a tensor of dimensions len(sequence) * MAX_LEN.
    """

    sequence = [x[:MAX_LEN] for x in sequence]    # Truncate to MAX_LEN
    
    first_padding = MAX_LEN - len(sequence[0])
    sequence[0] = sequence[0] + [0] * first_padding   # Pad first entry to MAX_LEN

    sequence = [torch.tensor(x) for x in sequence]     # Turn list into tensors
    sequence = pad_sequence(sequence, batch_first=True)     # Pad all lists of tensors to length of the largest (MAX_LEN)

    if ENABLE_CUDA: sequence = sequence.cuda()

    return sequence

def initialise_optimiser():
    """ Create optimiser with prespecified hyperparameters and return it. """

    # AdamW used to be BertAdam 
    max_grad_norm = 1.0
    num_total_steps = 1000
    num_warmup_steps = 100

    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    scheduler = transformers.get_linear_schedule_with_warmup(optimiser, num_warmup_steps, num_total_steps)

    return optimiser, scheduler

def training_epoch():
    """
    Perform one training epoch. Classify training data in batches, optimise model based on the loss.
    """ 
    model.train()
    total_loss, training_steps = 0, 0
    for index, batch in enumerate(train_dataloader):
        batch_input_ids, batch_attention_masks, batch_labels, batch_token_type_ids = batch      # Unpack input from dataloader
        print('Training batch {} with size {}.'.format(index, batch_labels.size()[0]))
        optimiser.zero_grad()       # Clear gradients
        model_output = model(batch_input_ids, token_type_ids=batch_token_type_ids, attention_mask=batch_attention_masks, labels=batch_labels)    # Forward pass
        logits = model_output[1]
        loss = criterion(logits, batch_labels)      # Calculate loss with class weights

        loss.backward()     # Backward pass 
        optimiser.step()
        scheduler.step()

        total_loss += loss.item()
        training_steps += 1
        train_loss.append(loss.item())

    print("Train loss: {}".format(total_loss / training_steps))
    # return loss.item()

def validation_epoch():
    """ Perform one validation epoch. Classify validation data in batches, calculate f1 scores. """
    model.eval()

    epoch_labels = torch.LongTensor()
    epoch_logits = torch.FloatTensor()

    for index, batch in enumerate(validation_dataloader): 
        batch_input_ids, batch_attention_masks, batch_labels, batch_token_type_ids = batch      # Unpack input from dataloader
        print('Validation batch {} with size {}.'.format(index, batch_labels.size()[0]))

        with torch.no_grad():       # Not computing gradients, saving memory and time 
            model_output = model(batch_input_ids, token_type_ids=batch_token_type_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            logits = model_output[1]

            # Save batch data to be able to evaluate epoch as a whole
            epoch_labels = torch.cat((epoch_labels, batch_labels.cpu()))
            epoch_logits = torch.cat((epoch_logits, logits.cpu()))

    micro_f1, macro_f1, c_matrix, preds = evaluate_model(epoch_logits, epoch_labels)
    
    return micro_f1, macro_f1, c_matrix
        
def train_model():

    for epoch in trange(NUM_EPOCHS, desc="Epoch"):
        training_epoch()
        micro_f1, macro_f1, c_matrix = validation_epoch()
    
    model.save_pretrained(MODEL_SAVE_DIR)
    return micro_f1, macro_f1, c_matrix

def evaluate_model(logits, epoch_labels):
    """ Use F1-score and confusion matrix to evaluate model performance. Logits and label tensors need to be on CPU. """
    preds = np.argmax(logits, axis=1).flatten()     # This gives us the flat predictions
    
    # F1 Score
    micro_f1 = f1_score(epoch_labels, preds, average='micro')
    macro_f1 = f1_score(epoch_labels, preds, average='macro')

    # Confusion matrix
    c_matrix = confusion_matrix(epoch_labels.tolist(), preds.tolist(), labels=[0, 1, 2])
    
    if(ENABLE_PLOTTING): plot_confusion_matrix(c_matrix)

    print('Micro f1: {}'.format(micro_f1))
    print('Macro f1: {}'.format(macro_f1))
    print(c_matrix)

    return micro_f1, macro_f1, c_matrix, preds

def plot_confusion_matrix(c_matrix):
    df_cm = pd.DataFrame(c_matrix, range(NUM_LABELS), range(NUM_LABELS))
    df_cm.columns = ['R', 'S', 'NEI']
    df_cm['Labels'] = ['R', 'S', 'NEI']
    df_cm = df_cm.set_index('Labels')
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.0)  # for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={'size':10})    # font size
    plt.show()

def plot_loss(train_loss):
    """ Plot loss after training the model"""
    plt.figure(figsize=(15,8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss)
    plt.show()

def export_training_results():
    results = {
        'config': {
            'train_data_path': TRAIN_DATA_PATH, 
            'validation_data_path': VALIDATION_DATA_PATH, 
            'test_data_path': TEST_DATA_PATH,
            'data_sample': DATA_SAMPLE,
            'pretrained_model': PRETRAINED_MODEL, 
            'max_len': MAX_LEN, 
            'batch_size': BATCH_SIZE, 
            'num_labels': NUM_LABELS, 
            'learning_rate': LEARNING_RATE, 
            'skip_training': ENABLE_TRAINING,
            'num_epochs': NUM_EPOCHS, 
            'enable_cuda': ENABLE_CUDA
        }, 
        'loss': train_loss, 
        'micro_f1': micro_f1, 
        'macro_f1': macro_f1, 
        'confusion_matrix': c_matrix.tolist()
    }
    filename = "train_{}.json".format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    output_path = OUTPUT_DIR + filename 
    with open(output_path, mode='w') as outfile:
        json.dump(results, outfile)

def test_model():
    """ Perform the test. Classify test data in batches, calculate f1 scores. """
    model.eval()

    test_labels = torch.LongTensor()
    test_logits = torch.FloatTensor()

    for index, batch in enumerate(test_dataloader): 
        batch_input_ids, batch_attention_masks, batch_labels, batch_token_type_ids = batch      # Unpack input from dataloader
        print('Test batch {} with size {}.'.format(index, batch_labels.size()[0]))

        with torch.no_grad():       # Not computing gradients, saving memory and time 
            model_output = model(batch_input_ids, token_type_ids=batch_token_type_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            logits = model_output[1]

            # Save batch data to be able to evaluate test as a whole
            test_labels = torch.cat((test_labels, batch_labels.cpu()))
            test_logits = torch.cat((test_logits, logits.cpu()))

    micro_f1, macro_f1, c_matrix, preds = evaluate_model(test_logits, test_labels)
    
    return micro_f1, macro_f1, c_matrix, preds

def export_test_results(df, preds):
    df['prediction'] = preds
    df = convert_labels(df, ['label', 'prediction'], reverse=True)
    filename = "test_{}.csv".format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    output_path = OUTPUT_DIR + filename 
    test_data_df.to_csv(OUTPUT_DIR + filename, sep=';')

    # Print to console
    print("\nTest results:")
    for index, row in test_data_df.iterrows():
        print("\nClaim:")
        print(row.claim)
        print("\nPrediction:")
        print(row.prediction)
        print("\nCorrect?")
        print(str(row.label == row.prediction))

#--------
# Main
#--------

start_time = datetime.now()

print('Reading config...')
CONFIG_PATH = 'config.json'
TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH, PRETRAINED_MODEL, MAX_LEN, BATCH_SIZE, NUM_LABELS, LEARNING_RATE, NUM_EPOCHS, ENABLE_PLOTTING, OUTPUT_DIR, ENABLE_TRAINING, DATA_SAMPLE, ENABLE_CUDA, ENABLE_TEST, MODEL_SAVE_DIR, ENABLE_TEST_EXPORT = read_config()
check_required_paths()
if ENABLE_CUDA: print('Using CUDA on {}.'.format(torch.cuda.get_device_name(0)))
# Use most recent .jsonl file in current directory if no test path was specified to be compatible with end-to-end process
if ENABLE_TEST and TEST_DATA_PATH == "":
    jsonl_files = glob.glob('out/*.jsonl')
    if len(jsonl_files):
        TEST_DATA_PATH = max(jsonl_files, key=os.path.getctime)
        print('No test path found. Using {}.'.format(TEST_DATA_PATH))
    else:
        print('No test path specified and no .jsonl files found in current directory! Aborting.')
        quit()
print('Reading config complete.')

print('Reading data...')
if ENABLE_TRAINING:
    train_data_df = pd.read_json(TRAIN_DATA_PATH, lines=True)
    validation_data_df = pd.read_json(VALIDATION_DATA_PATH, lines=True)
    train_data_df = train_data_df.head(int(DATA_SAMPLE * len(train_data_df.claim)))     # For sampling consistently 
    validation_data_df = validation_data_df.head(int(DATA_SAMPLE * len(validation_data_df)))
if ENABLE_TEST:
    test_data_df = pd.read_json(TEST_DATA_PATH, lines=True)
print('Reading data complete.')

print('Pre-processing data...')
if ENABLE_TRAINING:
    train_data_df = drop_duplicate_claims(train_data_df)
    validation_data_df = drop_duplicate_claims(validation_data_df)
    train_data_df = concatenate_evidence(train_data_df)
    validation_data_df = concatenate_evidence(validation_data_df)
    train_data_df = convert_labels(train_data_df)
    validation_data_df = convert_labels(validation_data_df)
# Test data evidence already comes concatenated from our document retrieval
if ENABLE_TEST:
    test_data_df = convert_labels(test_data_df)
    test_data_df.claim = test_data_df.claim.apply(lambda x: ftfy.fix_text(x))
    test_data_df.evidence = test_data_df.evidence.apply(lambda x: ftfy.fix_text(x))
print('Pre-processing complete.')

if ENABLE_TRAINING:
    print('Balancing data...')
    train_data_df, criterion = balance_data(train_data_df)
    print('Balancing data complete. Size of train data df: {}'.format(len(train_data_df.label)))

print('Preparing data...')
if ENABLE_TRAINING:
    train_inputs, train_token_type_ids = tokenize_inputs(train_data_df)
    validation_inputs, validation_token_type_ids = tokenize_inputs(validation_data_df)
    train_masks = generate_attention_masks(train_inputs)
    validation_masks = generate_attention_masks(validation_inputs)
    train_labels = train_data_df.label.tolist()
    validation_labels = validation_data_df.label.tolist()
if ENABLE_TEST: 
    test_inputs, test_token_type_ids = tokenize_inputs(test_data_df)
    test_masks = generate_attention_masks(test_inputs)
    test_labels = test_data_df.label.tolist()
print('Preparing data complete.')

print('Initialising dataloader...')
if ENABLE_TRAINING:
    train_dataloader = initialise_dataloader(train_inputs, train_masks, train_labels, train_token_type_ids)
    validation_dataloader = initialise_dataloader(validation_inputs, validation_masks, validation_labels, validation_token_type_ids)
if ENABLE_TEST: test_dataloader = initialise_dataloader(test_inputs, test_masks, test_labels, test_token_type_ids)
print('Initialising dataloader complete.')

print('Initialising model...')
model = initialise_model()
optimiser, scheduler = initialise_optimiser()
print('Initialising model complete.')

if ENABLE_TRAINING:
    print('Training model...')
    train_loss = []
    micro_f1, macro_f1, c_matrix = train_model()
    print('Training model complete.')

print('Execution time: {}.'.format(datetime.now()-start_time))

if ENABLE_TRAINING:
    print('Plotting loss...')
    plot_loss(train_loss)
    print('Plotting loss complete.')

    print('Exporting training results to json...')
    export_training_results()
    print('Export complete.')

if ENABLE_TEST:
    print('Running test...')
    test_micro_f1, test_macro_f1, test_c_matrix, test_preds = test_model()
    print('Test complete.')

    if ENABLE_TEST_EXPORT:
        print('Exporting test results to csv...')
        export_test_results(test_data_df, test_preds)
        print('Export complete.')