import torch
from typing import Any
from transformers import BertTokenizer, BertModel
from pytorch_lightning import LightningModule


EXAMPLES = ['테스트를 해볼까요?', '테스트가 잘 되네요~']


def batch_tokenize(tokenizer: BertTokenizer, list_of_texts: list[str]):
    return tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=list_of_texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )


class ModelForServiceBackend(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_model = BertModel.from_pretrained('klue/bert-base')
        self.pooler = lambda output: output.hidden_states[-1][:, 0, :]
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attention_mask, *args: Any, **kwargs: Any) -> Any:
        self.eval()
        with torch.no_grad():
            encoder_outputs = self.pretrained_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            pooled_outputs = self.pooler(encoder_outputs)
            output_logits = self.classifier(pooled_outputs)
            output_probs = torch.softmax(output_logits, dim=-1)
            # output_scores = output_probs[:, 1].squeeze()
            output_labels = torch.argmax(batch_probs, dim=-1)
            torch.cuda.empty_cache()
            return output_labels



if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
    print(tokenizer)

    model = ModelForServiceBackend.load_from_checkpoint(
        checkpoint_path='../checkpoints/YOUR_AWESOME_CHECKPOINT.ckpt',
        map_location={'cuda:0': 'cuda:0'}
    )
    print(model)

    tokenized = batch_tokenize(tokenizer, EXAMPLES).to('cuda')
    input_ids, token_type_ids, attention_mask = tokenized.values()
    print(tokenized)

    original_output = model(input_ids, token_type_ids, attention_mask)
    print(original_output)

    model.to_torchscript(
        file_path='your_awesome_service/1/model.pt',
        method='trace',
        example_inputs=[input_ids, attention_mask, token_type_ids],
    )

    loaded_model = torch.jit.load('your_awesome_service/1/model.pt').to('cuda')
    script_output = loaded_model(input_ids, token_type_ids, attention_mask)
    print(script_output)

    print(original_output - script_output)
