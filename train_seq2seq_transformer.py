import torch
import yaml
from src.models import trainer
from src.data.datamodule import DataManager
from src.txt_logger import TXTLogger
from src.models.seq2seq_transformer import Seq2SeqTransformer


def main(prin=False):
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    data_config = yaml.load(open("configs/data_config.yaml", 'r'), Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()

    model_config = yaml.load(open("configs/model_config.yaml", 'r'), Loader=yaml.Loader)

    # TODO: Инициализируйте модель Seq2SeqTransformer
    model = Seq2SeqTransformer(device=DEVICE,
                               num_encoder_layers=model_config["num_layers"],
                               emb_size=model_config["embedding_size"],
                               nhead=model_config["nhead"],
                               vocab_size=len(dm.tokenizer),
                               target_tokenizer=dm.target_tokenizer,
                               dim_feedforward=model_config["dim_feedforward"],
                               lr=model_config["learning_rate"]
                               ).to(DEVICE)

    logger = TXTLogger('training_logs')
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger, prin=prin)

    if model_config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)

    return model, dm, train_dataloader, dev_dataloader


if __name__ == "__main__":
    model, dm, train, val = main(True)
