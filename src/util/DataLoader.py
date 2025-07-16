from datasets import load_dataset, DatasetDict
from typing import cast
from torchtext.data import Field, Dataset, Example, BucketIterator
from Config import *
import logging


class DataLoader:
    
    def __init__(self, tokenize_de, tokenize_en, ext, init_token='<sos>', eos_token='<eos>'):
        """
        tokenize_de: 德语分词器
        tokenize_en: 英语分词器
        ext: 源语言和目标语言的扩展名 ('.de', '.en')
        init_token: 初始标记
        eos_token: 结束标记
        """
        self.tokenize_de = tokenize_de
        self.tokenize_en = tokenize_en
        self.ext = ext
        self.init_token = init_token
        self.eos_token = eos_token
        logging.info('dataset initializing start')

    def load_dataset_wmt14(self):
        wmt14_path = dataset_path + "wmt14/"
        dataset = load_dataset("parquet", data_files={
            "train": [f"{wmt14_path}/train-00000-of-00003.parquet",
                    f"{wmt14_path}/train-00001-of-00003.parquet",
                    f"{wmt14_path}/train-00002-of-00003.parquet"],
            "validation": f"{wmt14_path}/validation-00000-of-00001.parquet",
            "test": f"{wmt14_path}/test-00000-of-00001.parquet"
        })
        return dataset

    def load_dataset_multi30k(self):
        multi30k_path = dataset_path + "multi30k/"
        dataset = load_dataset("json", data_files={
            "train": f"{multi30k_path}/train.jsonl",
            "validation": f"{multi30k_path}/val.jsonl",
            "test": f"{multi30k_path}/test.jsonl"
        })
        return dataset

    def make_dataset(self):
        # dataset = self.load_dataset_wmt14()
        dataset = self.load_dataset_multi30k()
        dataset = cast(DatasetDict, dataset)

        self.source = Field(tokenize=self.tokenize_de, lower=True, init_token=self.init_token, eos_token=self.eos_token, batch_first=True)
        self.target = Field(tokenize=self.tokenize_en, lower=True, init_token=self.init_token, eos_token=self.eos_token, batch_first=True)
        
        train_data, validation_data, test_data = dataset["train"], dataset["validation"], dataset["test"]

        logging.debug("check train_data:")
        logging.debug(f"type(train_data): {type(train_data)}")
        logging.debug(f"train_data[0]: {train_data[0]}")

        train_data = self.huggingface_dataset_to_torchtext_dataset(train_data)
        validation_data = self.huggingface_dataset_to_torchtext_dataset(validation_data)
        test_data = self.huggingface_dataset_to_torchtext_dataset(test_data)

        return train_data, validation_data, test_data

    def huggingface_dataset_to_torchtext_dataset(self, dataset):
        """
        Args:
            dataset: Huggingface 数据集
        Returns:
            dataset: Torchtext 数据集
        """
        fields = [("src", self.source), ("trg", self.target)]
        examples = []
        for item in dataset:
            item = cast(dict, item)
            ext_src, ext_trg = self.ext
            # 去掉 "."
            ext_src = ext_src.replace('.', '')
            ext_trg = ext_trg.replace('.', '')
            # ex = Example.fromlist([item["translation"][ext_src], item["translation"][ext_trg]], fields)
            ex = Example.fromlist([item[ext_src], item[ext_trg]], fields)
            examples.append(ex)
        dataset = Dataset(examples, fields)
        return dataset

    def build_vocab(self, dataset, min_freq):
        """
        Args:
            dataset: Torchtext 数据集
            min_freq: 最小词频
        """
        self.source.build_vocab(dataset, min_freq=min_freq)
        self.target.build_vocab(dataset, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        """
        Args:
            train: 训练数据集
            validate: 验证数据集
            test: 测试数据集
            batch_size: 批量大小
            device: 设备
        Returns:
            train_iterator: 训练数据集迭代器
            valid_iterator: 验证数据集迭代器
            test_iterator: 测试数据集迭代器
        """
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              sort_key=lambda x: len(x.src),
                                                                              batch_size=batch_size,
                                                                              device=device)
        logging.info('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator