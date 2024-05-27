import torch
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, dropout=False, maxlen=5000):
        """
        emb_size - размер эмбеддингов
        maxlen - длинна контекста
        """
        super(PositionalEncoding, self).__init__()
        # TODO: Реализуйте конструтор https://pytorch.org/tutorials/beginner/translation_transformer.html
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        """
        token_embedding - тензор матрицы эмбеддингов
        """
        # TODO: Реализуйте сложение эмбединнгов токенов с позиционными эмбеддингами
        token_embedding = token_embedding * math.sqrt(token_embedding.size(-1))
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
