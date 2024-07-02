import glob
from collections.abc import Sequence
from typing import Iterable, Optional
import torch
from absl import app, flags
from lit_nlp import dev_server, server_flags
from lit_nlp.api import types
from lit_nlp.api.dataset import Dataset, Spec
from lit_nlp.api.model import Model, JsonDict
from supervision.modeling.sentence_classifier_kit import SentenceClassificationConfig, SentenceClassificationModel
from utility.json import load_jsons

FLAGS = flags.FLAGS
FLAGS.set_default("host", '0.0.0.0')
FLAGS.set_default("port", '4321')
FLAGS.set_default("development_demo", False)

LABELS = ['negative', 'positive']


class LITDataset(Dataset):
    def __init__(self, filename: str):
        super().__init__()
        filenames = glob.glob(filename)
        assert filenames, f'cannot load any files from "{filename}"'
        _loaded = load_jsons(filenames, encoding='utf-8', flatten=True)
        self._examples = [dict(uid=i['uid'], text=i['text'], label=i['label']) for i in _loaded]

    def spec(self) -> Spec:
        return {
            'uid': types.TextSegment(),
            'text': types.TextSegment(),
            'label': types.CategoryLabel(vocab=LABELS)
        }


class LITModel(Model):
    def __init__(self, pretrained_model_name_or_path: str, checkpoint_path: str, batch_size: int = 512):
        config = SentenceClassificationConfig(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            num_classes=2
        )
        # config.pretrained_model_max_seq_length = 512
        self._model = SentenceClassificationModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location={'cuda:0': 'cuda:0'},
            **{'config': config}
        )
        self.batch_size = batch_size

    def predict(self, inputs: Iterable[JsonDict], **kw) -> Iterable[JsonDict]:
        # batch processing
        self._model.eval()
        n = self.batch_size
        input_chunks = [inputs[i * n:(i + 1) * n] for i in range((len(inputs) + n - 1) // n)]
        outputs = []
        for batch in input_chunks:
            tokens = [' '.join(self._model.pretrained_tokenizer(i['text'], add_special_tokens=True)) for i in batch]
            with torch.no_grad():
                batch_inputs = self._model.tokenize(list_of_texts=[i['text'] for i in batch])
                encoder_outputs = self._model.pretrained_model(
                    **batch_inputs,
                    return_dict=True,
                    output_hidden_states=True
                )
                pooler_outputs = self._model.pooler(encoder_outputs)
                probs = self._model.config.activation(self._model.classifier(pooler_outputs)).cpu().numpy()
                embeddings = pooler_outputs.cpu().numpy()
            for input_id, (tok, emb, prob) in enumerate(zip(tokens, embeddings, probs)):
                out = dict(probs=prob, embeddings=emb, tokens=tok)
                outputs.append(out)
            torch.cuda.empty_cache()
        return outputs

    def input_spec(self) -> types.Spec:
        return {'text': types.TextSegment()}

    def output_spec(self) -> dict[str, types.LitType]:
        spec = dict(
            probs=types.MulticlassPreds(vocab=LABELS, parent='label'),
            embeddings=types.Embeddings(),       # <float>[emb_dim]: cls vector
            tokens=types.Tokens(parent='text'),  # list[str]
        )
        return spec

def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
    datasets = {
        'demo-dataset-01': LITDataset(filename='data/corpus/test/demo-01.json'),
        'demo-dataset-02': LITDataset(filename='data/corpus/test/demo-02.json'),
        'demo-dataset-03': LITDataset(filename='data/corpus/test/demo-03.json'),
    }

    models = {
        'demo-model-01': LITModel(
            pretrained_model_name_or_path='klue/bert-base',
            checkpoint_path='hypervision/checkpoints/bert-base.ckpt'
        ),
        'demo-model-02': LITModel(
            pretrained_model_name_or_path='klue/roberta-large',
            checkpoint_path='hypervision/checkpoints/roberta-large.ckpt'
        ),
    }

    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    return lit_demo.serve()


if __name__ == "__main__":
    # python -m interpretation.lit
    app.run(main)
