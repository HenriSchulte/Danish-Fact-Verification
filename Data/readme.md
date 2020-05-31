We use two versions of the data: 

1. The data in JSONL format contains all features:
	- claim: The claim to be verified
	- entity: List of page titles from which the evidence originates
	- evidence: List of evidence sentences
	- label: Supported, Refuted or NotEnoughInfo
	
2. The data in TSV format is used for evidence retrieval. It only consists of claims and labels.
	Training requires all features, therefore, no train set exists of this type.


Additionally, we also publish the parsed Danish wikipedia dataset that we use as a basis for data generation.
