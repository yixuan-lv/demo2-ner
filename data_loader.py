# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from utils import load_bio_data


class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, label2id, max_len=128):
        self.sentences, self.labels = load_bio_data(data_path)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

        # 预处理所有样本
        self.input_ids_list = []
        self.attention_masks_list = []
        self.label_ids_list = []

        for words, tags in zip(self.sentences, self.labels):

            encoding = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_len,
                padding="max_length"
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            word_ids = encoding.word_ids()

            label_ids = []

            for word_id in word_ids:

                if word_id is None:
                    label_ids.append(label2id['O'])

                else:
                    label_ids.append(
                        label2id[tags[word_id]]
                    )

            self.input_ids_list.append(input_ids)
            self.attention_masks_list.append(attention_mask)
            self.label_ids_list.append(label_ids)


    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids_list[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks_list[idx], dtype=torch.long),
            'labels': torch.tensor(self.label_ids_list[idx], dtype=torch.long)
        }


def create_dataloader(data_path, tokenizer, label2id, batch_size, max_len=128, shuffle=True):
    dataset = NERDataset(data_path, tokenizer, label2id, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)