import torch.nn as nn
import torch
import math

import src.metrics as metrics
from src.models.positional_encoding import PositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 emb_size,
                 nhead,
                 vocab_size,
                 device,
                 target_tokenizer,
                 dim_feedforward=512,
                 sched_step=5,
                 sched_gamma=0.1,
                 lr=0.001):
        super(Seq2SeqTransformer, self).__init__()
        # TODO: Реализуйте конструктор seq2seq трансформера - матрица эмбеддингов, позиционные эмбеддинги, encoder/decoder трансформер, vocab projection head
        # https://pytorch.org/tutorials/beginner/translation_transformer.html
        # https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

        self.device = device
        self.emb_size = emb_size
        self.target_tokenizer = target_tokenizer

        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward)
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size)
        self.decoder = nn.Linear(emb_size, vocab_size)

        self.loss = nn.CrossEntropyLoss()  # ignore_index=3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sched_step, gamma=sched_gamma, )

        self.src_mask = self.generate_square_subsequent_mask(0, 1).to(device)
        self.trg_mask = self.generate_square_subsequent_mask(0, 0).to(device)

    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
    def generate_square_subsequent_mask(self, sz, flag):
        if flag == 0:
            return torch.triu(torch.full((sz, sz), float('-inf'), device=self.device), diagonal=1)
        else:
            return torch.triu(torch.full((sz, sz), 0., device=self.device), diagonal=1)

    # https://pytorch.org/tutorials/beginner/translation_transformer.html
    def forward(self, batch):
        # TODO: Реализуйте forward pass для модели, при необходимости реализуйте другие функции для обучения
        src, trg = batch
        #trg = torch.cat([torch.tensor([[0]] * trg.size(0), device=self.device), trg], dim=1)[:, :-1]
        trg = trg[:, :-1]

        if self.src_mask.size(0) != src.size(1):
            self.src_mask = self.generate_square_subsequent_mask(src.size(1), 1)
        if self.trg_mask.size(0) != trg.size(1):
            self.trg_mask = self.generate_square_subsequent_mask(trg.size(1), 0)

        src = self.pos_encoder(self.embedding(src) * math.sqrt(self.emb_size)).permute(1, 0, 2)
        trg = self.pos_encoder(self.embedding(trg) * math.sqrt(self.emb_size)).permute(1, 0, 2)

        output = self.transformer(src, trg, self.src_mask, self.trg_mask)
        output = self.decoder(output).permute(1, 0, 2)
        topi = torch.argmax(output, dim=-1)

        return topi.clone(), output

    def training_step(self, batch):
        # TODO: Реализуйте обучение на 1 батче данных по примеру seq2seq_rnn.py
        self.optimizer.zero_grad()
        X_tensor, Y_tensor = batch
        predicted, decoder_outputs = self.forward(batch)
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        loss = self.loss(decoder_outputs, Y_tensor[:, 1:].reshape(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, batch):
        # TODO: Реализуйте оценку на 1 батче данных по примеру seq2seq_rnn.py
        X_tensor, Y_tensor = batch
        predicted, decoder_outputs = self.forward(batch)
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        loss = self.loss(decoder_outputs, Y_tensor[:, 1:].reshape(-1))
        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = predicted.squeeze(-1).detach().cpu().numpy()
        actuals = target_tensor.squeeze(-1).detach().cpu().numpy()
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences

    def predict(self, src_data, max_len=34):
        #  https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        trg_data = torch.ones(src_data.size(0), 1).fill_(0).type(torch.long).to(self.device)

        for i in range(max_len):
            src = src_data.clone()
            trg = trg_data.clone()
            if self.src_mask.size(0) != src.size(1):
                self.src_mask = self.generate_square_subsequent_mask(src.size(1), 1)
            if self.trg_mask.size(0) != trg.size(1):
                self.trg_mask = self.generate_square_subsequent_mask(trg.size(1), 0)

            src = self.pos_encoder(self.embedding(src) * math.sqrt(self.emb_size)).permute(1, 0, 2)
            trg = self.pos_encoder(self.embedding(trg) * math.sqrt(self.emb_size)).permute(1, 0, 2)

            output = self.transformer(src, trg, self.src_mask, self.trg_mask)
            output = self.decoder(output[-1, :, :])

            topi = torch.argmax(output, dim=-1)
            trg_data = torch.cat((trg_data, topi.reshape((-1, 1))), dim=1)

        return trg_data.clone()
