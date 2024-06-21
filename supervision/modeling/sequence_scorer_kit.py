from typing import Any, Literal
import torch
from supervision.modeling.base.config import ModelConfigBase
from supervision.modeling.base.model import LightningModuleBase


class SequenceScoringConfig(ModelConfigBase):
    def __init__(self, pretrained_model_name_or_path: str, batch_size: int = 32, learning_rate: float = 1e-5,
                 pooling_strategy: Literal['cls', 'mean', 'max', 'pooler_output'] = 'cls',
                 additional_special_tokens: list[str] = None):
        super().__init__(pretrained_model_name_or_path, additional_special_tokens)

        # pre-determined hyper-params of pretrained model
        self.pretrained_model_hidden_size = self.pretrained_config.hidden_size
        self.pretrained_model_max_token_length = self.pretrained_config.max_position_embeddings
        
        # Notes: these values are will be determined right before training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pooling_strategy = pooling_strategy
        self.objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.activation = torch.nn.Sigmoid()

        # optimizer and scheduler
        self.optimizer = torch.optim.AdamW
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        self.lr_scheduler_params = {
            'CosineAnnealingWarmRestarts': {"T_0": 500}
        }[self.lr_scheduler.__name__] if self.lr_scheduler else None

        # if you want to save pretrained things:
        # self.pretrained_tokenizer.save_pretrained('./pretrained_tokenizer/')
        # self.pretrained_model.save_pretrained('./pretrained_encoder/')

    @property
    def pooler_func(self):
        if self.pooling_strategy == 'cls':
            return lambda output: output.hidden_states[-1][:, 0, :]
        elif self.pooling_strategy == 'mean':
            return lambda output: torch.mean(output.hidden_states[-1], dim=1)
        elif self.pooling_strategy == 'max':
            return lambda output: torch.max(output.hidden_states[-1], dim=1)
        elif self.pooling_strategy == 'pooler_output':
            return lambda output: output.pooler_output
        else:
            raise NotImplementedError


class SequenceScoringModel(LightningModuleBase):
    def __init__(self, config: SequenceScoringConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Notice that you already have self.pretrained_tokenizer & self.pretrained_model once you initialize it
        self.pooler = config.pooler_func
        self.scorer = torch.nn.Linear(
            in_features=config.pretrained_model_hidden_size,
            out_features=1
        ).to(self.device)

    def tokenize(self, list_of_texts: (list[str], list[tuple[str, str]])):
        encoded = self.pretrained_tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=list_of_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.config.pretrained_model_max_token_length,
            return_tensors='pt'
        ).to(self.device)
        return encoded

    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        encoder_outputs = self.pretrained_model(
            **inputs,
            return_dict=True,
            output_hidden_states=True
        )
        # do something more here
        pooled_outputs = self.pooler(encoder_outputs)
        output_logits = self.scorer(pooled_outputs)
        torch.cuda.empty_cache()
        return output_logits

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

    def predict(self, samples, return_mode: Literal['logit', 'score'] = 'score'):
        self.eval()
        with torch.no_grad():
            batch_inputs = self.tokenize(list_of_texts=samples['texts'])
            batch_logits = self.forward(inputs=batch_inputs)
            if return_mode == 'logit':
                return batch_logits
            elif return_mode == 'score':
                return self.config.activation(batch_logits).squeeze(-1)
            else:
                raise ValueError("return_mode: Literal['logit', 'score']")


if __name__ == '__main__':
    config = SequenceScoringConfig(
        pretrained_model_name_or_path='klue/bert-base',
        pooling_strategy='cls'
    )
    model = SequenceScoringModel(config)

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
