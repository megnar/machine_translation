from transformers import AutoTokenizer


class T5Tokenizer:
    def __init__(self, max_sent_len):
        self.max_sent_len = max_sent_len
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        return self.tokenizer(sentence, padding="max_length", max_length=self.max_sent_len,
                              truncation=True, return_token_type_ids=True)

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        predict = self.tokenizer.decode(token_list, skip_special_tokens=True)
        return ''.join(predict)

    def __len__(self):
        return len(self.tokenizer)

    def add_tokens(self, token_list):
        self.tokenizer.add_tokens(token_list)
