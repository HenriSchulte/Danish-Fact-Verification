The classifier can be run for either fine-tuning (training), testing or both. To change the configuration, edit the config.json file in this directory.

The config's items are explained below. Settings marked with * only affect fine-tuning. Elements marked with + only affect testing.

Disclaimer: Not all settings are compatible with each other, and incorrect settings will crash the program!

- max_len: Length of input sequences (in tokens)
- batch_size: Size of test batches, decrease in case of out-of-memory issues
- num_labels: Number of labels in the data, in our case always 3
- num_epochs*: Number of fine-tuning epochs
- learning_rate*: Learning rate for fine-tuning
- train_data_path*: Path to train data as jsonl
- validation_data_path*: Path to validation data as jsonl
- test_data_path+: Path to test data as jsonl
- enable_test: Whether to test the model on the test data (0/1)
- data_sample: Allows for random sampling of the data. 1 uses all provided data (100%), 0.5 uses (50%), etc.
- pretrained-model: Path to the directory of the pre-trained model or name of a BERT model (e.g. bert-base-multilingual-cased)
- model_save_dir*: Directory in which to save the fine-tuned model
- enable_plotting: Whether to enable plotting of loss and confusion matrix (0/1)
- enable_training: Whether to fine-tune the model (0/1)
- output_dir*: Directory in which to export training results
- enable_cuda: Whether to use CUDA for GPU-accelleration (0/1)
	
Currently, config.json is configured for fine-tuning a model on a Tesla K80. batch_size and/or max_len need to be reduced for GPUs with less memory.
