
from ..config import config
from ..utils import alloc_logger
from ..preprocessor import loader
import torch
import transformers as tfs
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self, classifier):
        pretrained_weights = "bert-base-uncased"
        self.tokenizer = tfs.BertTokenizer.from_pretrained(pretrained_weights)
        self.bert = tfs.BertModel.from_pretrained(pretrained_weights)
        self.classifier = classifier
        self.logger = alloc_logger("Model.log", Model)
    
    def strs_to_tensor(self, texts: "List[str]")->torch.IntTensor:
        total_tokens = torch.zeros(size=(len(texts), config.MODEL.VEC_LEN), dtype=torch.long)

        for i in range(len(texts)):
            text = texts[i]
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            for j in range(len(tokens)):
                total_tokens[i][j] = tokens[j]

        tokens = total_tokens
        attention_mask = torch.where(tokens != 0, 1, 0)
        with torch.no_grad():
            vec = self.bert(tokens, attention_mask)
        return vec[0]


        



if __name__ == "__main__":
    samples = []
    labels = []
    for info in loader.walk_csv(config.PATH.DATA_TRAIN_CSV):
        samples.append(info.content)
        labels.append()


    model = Model(None)
    vec = model.str_to_tensor(sample)
    print(vec.shape)
    print(type(vec))
