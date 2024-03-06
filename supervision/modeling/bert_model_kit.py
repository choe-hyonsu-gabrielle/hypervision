from typing import Any
import torch
from supervision.modeling.base.config import ModelConfigBase
from supervision.modeling.base.model import LightningModuleBase


class BertClassifierConfig(ModelConfigBase):
    def __init__(self, pretrained_model_name_or_path: str, num_classes: int = 2, batch_size: int = 32, learning_rate: float = 1e-5):
        super().__init__(pretrained_model_name_or_path)

        # pre-determined hyper-params of pretrained model
        self.pretrained_model_hidden_size = self.pretrained_config.hidden_size
        self.pretrained_model_max_token_length = self.pretrained_config.max_position_embeddings
        self.num_classes = num_classes

        # Notes: these values are will be determined right before training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.objective = torch.nn.CrossEntropyLoss()
        self.activation = torch.nn.Softmax(dim=-1)

        # optimizer and scheduler
        self.optimizer = torch.optim.AdamW
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        self.lr_scheduler_params = {
            'CosineAnnealingWarmRestarts': {"T_0": 5000}
        }[self.lr_scheduler.__name__] if self.lr_scheduler else None

        # if you want to save pretrained things:
        # self.pretrained_tokenizer.save_pretrained('./pretrained_tokenizer/')
        # self.pretrained_model.save_pretrained('./pretrained_encoder/')


class BertClassifierModel(LightningModuleBase):
    def __init__(self, config: BertClassifierConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Notice that you already have self.pretrained_tokenizer & self.pretrained_model once you initialize it
        self.pooler = lambda output: output.hidden_states[-1][:, 0, :]  # returns embedded [CLS] token vector
        # ... or you can use the function which returns mean of all token vectors
        # output.hidden_states[-1] is last_hidden_state of all tokens sized batch_size * L * hidden_size (or d_model)
        # self.pooler = lambda output: torch.mean(output.hidden_states[-1], dim=1)
        self.downstream_task_head = torch.nn.Linear(
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

    def forward(self, inputs, apply_activation=False, *args: Any, **kwargs: Any) -> Any:
        encoder_outputs = self.pretrained_model(
            **inputs,
            return_dict=True,
            output_hidden_states=True
        )
        # do something more here
        pooled_outputs = self.pooler(encoder_outputs)
        outputs = self.downstream_task_head(pooled_outputs)
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
        batch_logits = self.forward(inputs=batch_inputs, apply_activation=True)
        return torch.argmax(batch_logits, dim=-1)


if __name__ == '__main__':
    config = BertClassifierConfig(pretrained_model_name_or_path='klue/bert-base', num_classes=2)
    model = BertClassifierModel(config)

    samples = dict(texts=['테스트를 해볼까요?', '테스트가 잘 되네요~'], targets=[[1, 0], [0, 1]], labels=[0, 1])

    tokens_a = model.pretrained_tokenizer.tokenize(samples['texts'][0], add_special_tokens=True)
    tokens_b = model.pretrained_tokenizer.tokenize(samples['texts'][-1], add_special_tokens=True)
    print(tokens_a, tokens_b)
    # ['[CLS]', '테스트', '##를', '해볼', '##까요', '?', '[SEP]'] ['[CLS]', '테스트', '##가', '잘', '되네요', '~', '[SEP]']

    tokenized = model.pretrained_tokenizer.batch_encode_plus(samples['texts'], add_special_tokens=True)
    print(tokenized)
    # {'input_ids': [[2, 23825, 4180, 21951, 8986, 32, 3], [2, 23825, 4009, 2483, 25496, 95, 3]],
    #  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
    #  'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]}

    loss = model.batch_forward_and_loss(samples)
    print(loss)  # tensor(0.7254, grad_fn=<DivBackward1>)

    prediction = model.predict(samples)
    print(prediction)  # tensor([1, 1])
