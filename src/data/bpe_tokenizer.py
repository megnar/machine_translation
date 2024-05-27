from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import decoders

class BPETokenizer:
    def __init__(self, sentence_list, pad_flag, vocab_size, min_frequency, max_sent_len):
        """
        sentence_list - список предложений для обучения
        """
        # TODO: Реализуйте конструктор c помощью https://huggingface.co/docs/transformers/fast_tokenizers, обучите токенизатор, подготовьте нужные аттрибуты(word2index, index2word)
        self.pad_flag = pad_flag
        self.max_sent_len = max_sent_len+2
        self.tokenizer = Tokenizer(BPE(unk_token="UNK"))
        #self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.decoder = decoders.BPEDecoder(" ")

        self.special_tokens = ["SOS", "EOS", "UNK", "PAD"]
        trainer = BpeTrainer(special_tokens=self.special_tokens, vocab_size=vocab_size, min_frequency=min_frequency)
        self.tokenizer.train_from_iterator(sentence_list, trainer)

        #self.word2index = self.tokenizer.get_vocab()
        #self.index2word = {i: k for k, i in self.word2index.items()}

    def pad_sent(self, token_ids_list):
        if len(token_ids_list) < self.max_sent_len:
            padded_token_ids_list = token_ids_list + [self.tokenizer.token_to_id('PAD')] * (
                        self.max_sent_len - len(token_ids_list))
        else:
            padded_token_ids_list = token_ids_list[:self.max_sent_len - 1] + [self.tokenizer.token_to_id('EOS')]
        return padded_token_ids_list

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        # TODO: Реализуйте метод токенизации с помощью обученного токенизатора
        tokenized_data = [self.tokenizer.token_to_id('SOS')] + self.tokenizer.encode(sentence).ids + [self.tokenizer.token_to_id('EOS')]
        if self.pad_flag:
            tokenized_data = self.pad_sent(tokenized_data)

        return tokenized_data

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        # TODO: Реализуйте метод декодирования предсказанных токенов
        #words = list(filter(lambda x: x not in self.special_tokens, self.tokenizer.decode(token_list).split(' ')))
        return "".join(self.tokenizer.decode(token_list))

    def __len__(self):
        return len(self.tokenizer.get_vocab())
