import ast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
torch.manual_seed(0)
import pytreebank
from transformers import DistilBertTokenizerFast

DATASET_DIR = Path('data')
DATASET_DIR.mkdir(exist_ok=True)

SPLITS = ['train', 'dev', 'test']

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
pad_token_id = tokenizer.pad_token_id
mask_token_id = tokenizer.mask_token_id
sep_token_id = tokenizer.sep_token_id
cls_token_id = tokenizer.cls_token_id


class SentimentRationaleDataset(torch.utils.data.Dataset):
	def __init__(self, split, limit=None):
		super().__init__()
		self.split = split
		self.df = self._load_split(split)
		if limit is not None:
			self.df = self.df.sample(limit, random_state=0)
		self.df['text'] = self.df['text'].apply(lambda s: s.split())

	def _load_split(self, split):
		try:
			df = pd.read_csv(DATASET_DIR / f'{split}.tsv', sep='\t', converters={'rationale': ast.literal_eval})
		except FileNotFoundError as e:
			raise FileNotFoundError('You first need to run `python utils/dataset.py` to download and preprocess the dataset.') from e
		return df

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		label = 1 if row['label'] == 'pos' else 0
		return {'text': row['text'], 'rationale': row['rationale'], 'label': label}

	def get_dataloader(self, batch_size=8, shuffle=False):
		dl = torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=_custom_collate)
		return dl


def _custom_collate(batch):
	result = {k: [] for k in batch[0].keys()}
	for instance in batch:
		for k in instance.keys():
			result[k].append(instance[k])
	return result


def tokenize(texts, rationales=None):
	"""
	Adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb
	"""
	tokenized_inputs = tokenizer(text=texts, truncation=True, padding=True, is_split_into_words=True, return_tensors='pt')

	if rationales is not None:
		labels = []
		for i, label in enumerate(rationales):
			word_ids = tokenized_inputs.word_ids(batch_index=i)
			previous_word_idx = None
			label_ids = []
			for word_idx in word_ids:
				# Special tokens have a word id that is None. We set the label to -100 so they are automatically
				# ignored in the loss function.
				if word_idx is None:
					label_ids.append(-100)
				# We set the label for the first token of each word.
				elif word_idx != previous_word_idx:
					label_ids.append(label[word_idx])
				# For the other tokens in a word, we set the label to either the current label or -100, depending on
				# the label_all_tokens flag.
				else:
					label_ids.append(label[word_idx])# if label_all_tokens else -100)
				previous_word_idx = word_idx
			labels.append(label_ids)
		tokenized_inputs["rationales"] = torch.tensor(labels)
	return tokenized_inputs


def decode(input_ids):
	return tokenizer.decode(input_ids)


def download_and_preprocess_dataset():
	"""
	Preprocessing adapted from https://github.com/BoulderDS/evaluating-human-rationales/blob/66402dbe8ccdf8b841c185cd8050b8bdc04ef3d2/scripts/download_and_process_sst.py
	"""
	print('Preparing SST rationale dataset...')
	dataset = pytreebank.load_sst()
	for split in SPLITS:
		print(f'\tPreprocessing {split} split')
		dataset_split = dataset[split]
		instances = []
		threshold = 1.0
		for idx, instance in enumerate(dataset_split):
			instance_dict = {'id':idx}
			if instance.label < 2:
				instance_dict['label'] = "neg"
			elif instance.label > 2:
				instance_dict['label'] = "pos"
			elif instance.label == 2:
				continue #Skip neutral instances
			else:
				raise ValueError(f'unknown instance label: {instance.label}')

			instance_dict['original_text'] = []
			instance_dict['text'] = []
			instance_dict['rationale'] = []


			leaves = __get_leaves(instance)
			for leaf in leaves:
				instance_dict['original_text'].append(leaf.text)
				instance_dict['text'].append(leaf.text.lower())
			__count_leaves_and_extreme_descendants(instance)
			phrases = []
			__assemble_rationale_phrases(instance, phrases)
			for phrase in phrases:
				phrase_rationale = [np.abs(__normalize_label(phrase.label))] * phrase.num_leaves
				phrase_rationale = [0 if r < 0.5 else 1 for r in phrase_rationale]
				instance_dict['rationale'].extend(phrase_rationale)
			assert(len(instance_dict['rationale']) == len(instance_dict['text']))

			tokenization = []
			start, end = 0,0
			for token in instance_dict['original_text']:
				end = start + len(token)
				tokenization.append([start, end])
				start = end + 1

			instance_dict['tokenization'] = tokenization
			instance_dict['original_text'] = ' '.join(instance_dict['original_text'])
			instance_dict['text'] = ' '.join(instance_dict['text'])

			# take care of edge cases where a tokens spans across whitespace.
			tokens = [instance_dict['text'][start:end] for start, end in tokenization]
			tokens = [t.replace(' ', '-') for t in tokens]
			instance_dict['tokens'] = tokens
			instance_dict['text'] = ' '.join(instance_dict['tokens'])

			instances.append(instance_dict)

		dataset_split_df = pd.DataFrame(instances)
		dataset_split_df = dataset_split_df.drop(['id', 'original_text', 'tokenization'], axis=1)
		dataset_split_df['text'] = dataset_split_df['text'].apply(lambda s: s.replace('\t', ' '))

		print(f'Saving {split}.tsv with {dataset_split_df.shape[0]} rows in {DATASET_DIR}/.')
		dataset_split_df.to_csv(DATASET_DIR / f'{split}.tsv', sep='\t', index=False)


def __get_leaves(tree):
    leaves = []
    if len(tree.children) > 0:
        for child in tree.children:
            leaves += __get_leaves(child)
    else:
        leaves.append(tree)
    return leaves


def __count_leaves_and_extreme_descendants(tree):

	if len(tree.children) == 0: #if is leaf
		tcount = 1
		tmax = tmin = tree.label
	else:
		tcount = 0
		child_labels = [child.label for child in tree.children]
		tmax = max(child_labels)
		tmin = min(child_labels)

		for child in tree.children:
			ccount, cmax, cmin = __count_leaves_and_extreme_descendants(child)
			tcount += ccount
			tmax = max(tmax, cmax)
			tmin=min(tmin, cmin)

	tree.num_leaves=tcount
	tree.max_descendant = tmax
	tree.min_descendant = tmin

	if tree.label == 4:
		_=None
	return tcount, tmax, tmin


def __normalize_label(label):
	return (label-2)/2


def __explanatory_phrase(tree):
	if len(tree.children) == 0:
		return True
	else:
		#if this phrase is of extreme sentiment which is not explained by a descendant
		normalized_label = __normalize_label(tree.label)
		normalized_max_descendant = __normalize_label(tree.max_descendant)
		normalized_min_descendant = __normalize_label(tree.min_descendant)

		#if label is higher than highest descendant or lower than lowest descendant
		# if (normalized_label - normalized_max_descendant) > 0.5 or (normalized_label - normalized_min_descendant) < -0.5:
		if abs(normalized_label) > abs(normalized_max_descendant) and abs(normalized_label) > abs(normalized_min_descendant):
			return True
		else:
			return False


def __assemble_rationale_phrases(tree, phrases, **kwargs):
	if __explanatory_phrase(tree, **kwargs):
		phrases.append(tree)
	else:
		for child in tree.children:
			__assemble_rationale_phrases(child, phrases)


if __name__ == '__main__':
	download_and_preprocess_dataset()

	ds = SentimentRationaleDataset('train')
	dl = ds.get_dataloader()

	for batch in dl:
		i = 0
