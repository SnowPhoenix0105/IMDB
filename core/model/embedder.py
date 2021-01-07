
import torch
import numpy as np
import transformers as tfs
from ..config import config

class Embedder:
    def __init__(self):
        self.pretrained_weights = "bert-base-uncased"
        self.tokenizer = tfs.BertTokenizer.from_pretrained(self.pretrained_weights)
        self.bert = tfs.BertModel.from_pretrained(self.pretrained_weights).cuda()
        
    def strs_to_tensor(self, texts: "List[str]")->torch.Tensor:
        text_count = len(texts)
        # self.logger.log_message("text_count=", text_count)

        total_tokens = torch.zeros(size=(text_count, config.MODEL.VEC_LEN), dtype=torch.long)
        max_text_len = config.MODEL.VEC_LEN << 2

        for i in range(text_count):
            text = texts[i]
            if len(text) > max_text_len:
                text = text[:max_text_len]
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            for j in range(min(len(tokens), config.MODEL.VEC_LEN)):
                total_tokens[i][j] = tokens[j]

        # self.logger.log_message("finish tokenization")
        total_tokens = total_tokens.cuda()
        attention_mask = torch.where(total_tokens != 0, 1, 0)
        with torch.no_grad():
            vec = self.bert(total_tokens, attention_mask)
        # self.logger.log_message("finish word2vec")
        del total_tokens
        del attention_mask
        return vec[0].cpu()

    def strs_to_numpy(self, texts: "List[str]")->np.ndarray:
        samples_count = len(texts)
        X = self.strs_to_tensor(texts).numpy()
        shape = X.shape
        return X.reshape(shape[0], shape[1] * shape[2])