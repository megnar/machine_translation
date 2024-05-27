import torch

from transformers import T5ForConditionalGeneration
from transformers.optimization import Adafactor
import src.metrics as metrics


class Seq2SeqT5(torch.nn.Module):
    def __init__(self, size_token_embeddings, tokenizer, device, lr, sched_step=5, sched_gamma=0.1):
        super(Seq2SeqT5, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        # TODO: Реализуйте конструктор seq2seq t5 - https://huggingface.co/docs/transformers/model_doc/t5

        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
        self.model.resize_token_embeddings(size_token_embeddings)
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(torch.randn(param.size()))

        self.optimizer = Adafactor(self.model.parameters(), lr=lr, relative_step=False)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sched_step, gamma=sched_gamma)

    def forward(self, batch):
        input_tensor, target_tensor, attention_mask = batch
        # TODO: Реализуйте forward pass для модели, при необходимости реализуйте другие функции для обучения
        output = self.model(input_ids=input_tensor, attention_mask=attention_mask, labels=target_tensor)
        topi = torch.argmax(output.logits, dim=-1)
        return topi.clone(), output

    def training_step(self, batch):
        # TODO: Реализуйте обучение на 1 батче данных по примеру seq2seq_rnn.py
        self.model.train()
        self.optimizer.zero_grad()
        _, decoder_outputs = self(batch)
        loss = decoder_outputs.loss
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def validation_step(self, batch):
        # TODO: Реализуйте оценку на 1 батче данных по примеру seq2seq_rnn.py
        with torch.no_grad():
            _, decoder_outputs = self(batch)
            loss = decoder_outputs.loss
        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = predicted.squeeze(-1).detach().cpu().numpy()
        actuals = target_tensor.squeeze(-1).detach().cpu().numpy()
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences

    def predict(self, src_data, attention_mask, max_len=34):
        return self.model.generate(src_data, attention_mask=attention_mask, max_length=max_len)




