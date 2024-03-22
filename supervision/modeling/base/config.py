from transformers import AutoConfig, AutoTokenizer, AutoModel


class ModelConfigBase:
    def __init__(self, pretrained_model_name_or_path: str, additional_special_tokens: dict[str, str] = None):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.additional_special_tokens = additional_special_tokens
        self._pretrained_config = None
        self._pretrained_tokenizer = None
        self._pretrained_model = None

        # pre-determined hyper-params of pretrained model
        self.pretrained_model_hidden_size = None
        self.pretrained_model_max_token_length = None
        self.num_classes = None

        # Notes: these values are will be determined right before training
        self.batch_size = None
        self.learning_rate = None
        self.objective = None
        self.activation = None

        # optimizer and scheduler
        self.optimizer = None
        self.lr_scheduler = None
        self.lr_scheduler_params = None

        # if you want to save pretrained things:
        # self.pretrained_tokenizer.save_pretrained('./pretrained_tokenizer/')
        # self.pretrained_model.save_pretrained('./pretrained_encoder/')

    @property
    def pretrained_config(self):
        if self._pretrained_config is None:
            print(f'[{self.__class__.__name__}] loading pretrained config of `{self.pretrained_model_name_or_path}`')
            self._pretrained_config = AutoConfig.from_pretrained(self.pretrained_model_name_or_path)
        return self._pretrained_config

    @property
    def pretrained_tokenizer(self):
        if self._pretrained_tokenizer is None:
            print(f'[{self.__class__.__name__}] loading pretrained tokenizer of `{self.pretrained_model_name_or_path}`')
            self._pretrained_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
            if self.additional_special_tokens:
                print(f'[{self.__class__.__name__}] appending special tokens: {self.additional_special_tokens}')
                self._pretrained_tokenizer.add_special_tokens(self.additional_special_tokens)
        return self._pretrained_tokenizer

    @property
    def pretrained_model(self):
        if self._pretrained_model is None:
            print(f'[{self.__class__.__name__}] loading pretrained model of `{self.pretrained_model_name_or_path}`')
            self._pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name_or_path)
            if self.additional_special_tokens and self._pretrained_tokenizer is not None:
                print(f'[{self.__class__.__name__}] resizing token embeddings to: {len(self._pretrained_tokenizer)}')
                self._pretrained_model.resize_token_embeddings(len(self._pretrained_tokenizer))
        return self._pretrained_model
