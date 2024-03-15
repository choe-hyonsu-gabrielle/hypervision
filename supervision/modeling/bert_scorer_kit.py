from typing import Any
import torch
from supervision.modeling.base.config import ModelConfigBase
from supervision.modeling.base.model import LightningModuleBase


class BertScorerConfig(ModelConfigBase):
    def __init__(self, pretrained_model_name_or_path: str, batch_size: int = 32, learning_rate: float = 1e-5):
        super().__init__(pretrained_model_name_or_path)

        # pre-determined hyper-params of pretrained model
        self.pretrained_model_hidden_size = self.pretrained_config.hidden_size
        self.pretrained_model_max_token_length = self.pretrained_config.max_position_embeddings

        # Notes: these values are will be determined right before training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.activation = torch.nn.Sigmoid()

        # optimizer and scheduler
        self.optimizer = torch.optim.AdamW
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        self.lr_scheduler_params = {
            'CosineAnnealingWarmRestarts': {"T_0": 5000}
        }[self.lr_scheduler.__name__] if self.lr_scheduler else None

        # if you want to save pretrained things:
        # self.pretrained_tokenizer.save_pretrained('./pretrained_tokenizer/')
        # self.pretrained_model.save_pretrained('./pretrained_encoder/')


class BertScorerModel(LightningModuleBase):
    def __init__(self, config: BertScorerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Notice that you already have self.pretrained_tokenizer & self.pretrained_model once you initialize it
        self.pooler = lambda output: output.hidden_states[-1][:, 0, :]  # returns embedded [CLS] token vector
        # ... or you can use the function which returns mean of all token vectors
        # output.hidden_states[-1] is last_hidden_state of all tokens sized batch_size * L * hidden_size (or d_model)
        # self.pooler = lambda output: torch.mean(output.hidden_states[-1], dim=1)
        self.scorer = torch.nn.Linear(
            in_features=config.pretrained_model_hidden_size,
            out_features=1
        ).to(self.device)

    def tokenize(self, list_of_texts: list[str]):
        encoded = self.pretrained_tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=list_of_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.config.pretrained_model_max_token_length,
            return_tensors='pt'
        ).to(self.device)
        return encoded

    def forward(self, inputs, apply_activation=False, *args: Any, **kwargs: Any) -> Any:
        encoder_outputs = self.pretrained_model(
            **inputs,
            return_dict=True,
            output_hidden_states=True
        )
        # do something more here
        pooled_outputs = self.pooler(encoder_outputs)
        outputs = self.scorer(pooled_outputs)
        if apply_activation:
            outputs = self.config.activation(outputs)
        torch.cuda.empty_cache()
        return outputs

    def compute_loss(self, output_logit: torch.Tensor, target_logit: torch.Tensor):
        loss = self.config.objective(output_logit, target_logit)
        return loss

    def batch_forward_and_loss(self, samples):
        batch_inputs = self.tokenize(list_of_texts=samples['texts'])
        batch_logits = self.forward(inputs=batch_inputs)
        batch_loss = self.compute_loss(
            output_logit=batch_logits,
            target_logit=torch.tensor(samples['targets'], dtype=torch.float, device=self.device)
        )
        return batch_loss

    def predict(self, samples):
        self.eval()
        batch_inputs = self.tokenize(list_of_texts=samples['texts'])
        return self.forward(inputs=batch_inputs, apply_activation=True).squeeze(-1)


if __name__ == '__main__':
    config = BertScorerConfig(pretrained_model_name_or_path='klue/bert-base')
    model = BertScorerModel(config)

    samples = dict(texts=[('테스트를 해볼까요?', '테스트가 잘 되네요~'), ('1번', '2번')], targets=[[1], [0]], labels=[1, 0])

    tokens_a = model.pretrained_tokenizer.tokenize(samples['texts'][0], add_special_tokens=True)
    tokens_b = model.pretrained_tokenizer.tokenize(samples['texts'][1], add_special_tokens=True)
    print(tokens_a, tokens_b)
    # ['[CLS]', '테스트', '##를', '해', '##볼', '##까요', '?', '[SEP]', '테스트', '##가', '잘', '되', '##네', '##요', '~', '[SEP]']
    # ['[CLS]', '1', '##번', '[SEP]', '2', '##번', '[SEP]']

    tokenized = model.pretrained_tokenizer.batch_encode_plus(batch_text_or_text_pairs=samples['texts'], add_special_tokens=True)
    print(tokenized)
    # {'input_ids': [[2, 7453, 2138, 1897, 2345, 6301, 35, 3, 7453, 2116, 1521, 859, 2203, 2182, 97, 3], [2, 21, 2517, 3, 22, 2517, 3]],
    #  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1]],
    #  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]}

    loss = model.batch_forward_and_loss(samples)
    print(loss)  # tensor(1.1624, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)

    prediction = model.predict(samples)
    print(prediction)  # tensor([0.6793, 0.8560], grad_fn=<SqueezeBackward1>)
