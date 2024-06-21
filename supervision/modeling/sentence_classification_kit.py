from typing import Any, Literal
import torch
from supervision.modeling.base.config import ModelConfigBase
from supervision.modeling.base.model import LightningModuleBase


class SentenceClassificationConfig(ModelConfigBase):
    def __init__(self, pretrained_model_name_or_path: str, num_classes: int = 2, batch_size: int = 32,
                 learning_rate: float = 1e-5, pooling_strategy: Literal['cls', 'mean', 'max', 'pooler_output'] = 'cls',
                 additional_special_tokens: list[str] = None):
        super().__init__(pretrained_model_name_or_path, additional_special_tokens)

        # pre-determined hyper-params of pretrained model
        self.pretrained_model_hidden_size = self.pretrained_config.hidden_size
        self.pretrained_model_max_token_length = self.pretrained_config.max_position_embeddings
        self.num_classes = num_classes

        # Notes: these values are will be determined right before training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pooling_strategy = pooling_strategy
        self.objective = torch.nn.CrossEntropyLoss()
        self.activation = torch.nn.Softmax(dim=-1)

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


class SentenceClassificationModel(LightningModuleBase):
    def __init__(self, config: SentenceClassificationConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Notice that you already have self.pretrained_tokenizer & self.pretrained_model once you initialize it
        self.pooler = config.pooler_func
        self.classifier = torch.nn.Linear(
            in_features=config.pretrained_model_hidden_size,
            out_features=config.num_classes
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

    def forward(self, inputs, *args: Any, **kwargs: Any) -> Any:
        encoder_outputs = self.pretrained_model(
            **inputs,
            return_dict=True,
            output_hidden_states=True
        )
        # do something more here
        pooled_outputs = self.pooler(encoder_outputs)
        output_logits = self.classifier(pooled_outputs)
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

    def predict(self, samples, return_mode: Literal['logit', 'probs', 'score', 'label'] = 'label'):
        self.eval()
        with torch.no_grad():
            batch_inputs = self.tokenize(list_of_texts=samples['texts'])
            batch_logits = self.forward(inputs=batch_inputs)
            if return_mode == 'logit':
                return batch_logits
            elif return_mode == 'probs':
                batch_probs = self.config.activation(batch_logits)
                return batch_probs
            elif return_mode == 'score':
                batch_probs = self.config.activation(batch_logits)
                return batch_probs[:, 1].squeeze()
            elif return_mode == 'label':
                batch_probs = self.config.activation(batch_logits)
                return torch.argmax(batch_probs, dim=-1)
            else:
                raise ValueError("return_mode: Literal['logit', 'probs', 'score', 'label']")


if __name__ == '__main__':
    config = SentenceClassificationConfig(
        pretrained_model_name_or_path='klue/bert-base',
        num_classes=2,
        pooling_strategy='cls'
    )
    model = SentenceClassificationModel(config)

    samples = dict(texts=['테스트를 해볼까요?', '테스트가 잘 되네요~'], targets=[[1, 0], [0, 1]], labels=[0, 1])

    tokens_a = model.pretrained_tokenizer.tokenize(samples['texts'][0], add_special_tokens=True)
    tokens_b = model.pretrained_tokenizer.tokenize(samples['texts'][1], add_special_tokens=True)
    print(tokens_a, tokens_b)
    # ['[CLS]', '테스트', '##를', '해볼', '##까요', '?', '[SEP]'] ['[CLS]', '테스트', '##가', '잘', '되네요', '~', '[SEP]']

    tokenized = model.pretrained_tokenizer.batch_encode_plus(samples['texts'], add_special_tokens=True)
    print(tokenized)
    # {'input_ids': [[2, 23825, 4180, 21951, 8986, 32, 3], [2, 23825, 4009, 2483, 25496, 95, 3]],
    #  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
    #  'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]}

    loss = model.batch_forward_and_loss(samples)  # returns loss
    print(loss)  # tensor(0.6630, grad_fn=<DivBackward1>)

    prediction = model.predict(samples)  # returns labels
    print(prediction)  # tensor([1, 1])

    probs = model.predict(samples, return_mode='probs')
    print(probs)  # tensor([[0.3996, 0.6004], [0.3356, 0.6644]], grad_fn=<SoftmaxBackward0>)

    score = model.predict(samples, return_mode='score')
    print(score)  # tensor([0.6004, 0.6644], grad_fn=<SqueezeBackward0>)
