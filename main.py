"""Training entry point for the BERT-BiLSTM NER model."""

import logging
import sys

import swanlab
import torch
from transformers import AutoTokenizer

from config import Config
from data_loader import NERDataModule
from model import NERModel
from trainer import Trainer
from utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    dataset_name = "weibo"
    model_name = "chinese-bert-wwm"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config.from_json(f"configs/{dataset_name}_config.json")
    config.bert_path = f"./bert_models/{model_name}"
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    data = NERDataModule(
        data_paths=config.data_path[dataset_name],
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_seq_len,
    )
    data.setup()

    model = NERModel(
        config=config,
        num_labels=data.num_labels,
        id2label=data.id2label,
    ).to(device)
    logger.info(
        "Model parameters: total=%d, trainable=%d",
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )

    swanlab.init(
        project="demo2-ner",
        experiment_name=f"{dataset_name}_{model_name}_{config.bert_lr}",
        config={**config.__dict__, "dataset": dataset_name, "model": model_name},
    )
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        train_loader=data.train_dataloader(),
        dev_loader=data.dev_dataloader(),
        test_loader=data.test_dataloader(),
        id2label=data.id2label,
    )
    test_f1, test_report = trainer.train()
    logger.info("Final test F1: %.4f\n%s", test_f1, test_report)
    swanlab.finish()
    return test_f1, test_report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        sys.exit(0)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)
