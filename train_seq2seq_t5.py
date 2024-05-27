import torch
import yaml
from src.models import seq2seq_t5, trainer
from src.data.datamodule import DataManager
from src.txt_logger import TXTLogger


def main(prin=False):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    DEVICE = 'cpu'


    data_config = yaml.load(open("configs/data_config.yaml", 'r'), Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()

    model_config = yaml.load(open("configs/model_config.yaml", 'r'), Loader=yaml.Loader)

    # TODO: Инициализируйте модель Seq2SeqTransformer
    model = seq2seq_t5.Seq2SeqT5(size_token_embeddings=len(dm.tokenizer),
                                 tokenizer=dm.tokenizer,
                                 device=DEVICE,
                                 lr=model_config["learning_rate"])

    logger = TXTLogger('training_logs')
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger, prin=prin)

    if model_config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)

    return model, dm, train_dataloader, dev_dataloader


if __name__ == "__main__":
    model, dm, train, val = main(True)
