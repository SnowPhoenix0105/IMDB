
from ..config import config
from ..utils import alloc_logger
from ..preprocessor import loader
import torch
import numpy as np
import transformers as tfs
import shutil
import joblib
import os
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import PCA

class Model:
    def __init__(self, classifier=None):
        self.pretrained_weights = "bert-base-uncased"
        self.dim_reducer = None
        self.classifier = classifier if classifier is not None else BernoulliNB()
        self.logger = alloc_logger("Model.log", Model)

        self.logger.log_message("loading bert...")
        self.tokenizer = tfs.BertTokenizer.from_pretrained(self.pretrained_weights)
        self.bert = tfs.BertModel.from_pretrained(self.pretrained_weights).cuda()
        self.logger.log_message("finish loading!")
        self.logger.log_message("pca_dim=", config.MODEL.PCA_DIM)
    
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

    def partial_fit(self, batch_X, batch_y):
        # print(batch_X.shape)
        # print(batch_y.shape)
        if config.MODEL.PCA_DIM < 768:
            if self.dim_reducer is None:
                self.dim_reducer = PCA(config.MODEL.PCA_DIM)
                batch_X = self.dim_reducer.fit_transform(batch_X)
                self.logger.log_message("pca.explained_variance_ratio_=", self.dim_reducer.explained_variance_ratio_)
            else:
                batch_X = self.dim_reducer.transform(batch_X)
        self.classifier.partial_fit(batch_X, batch_y, classes=[0,1])


    def fit(self, X, out):
        signature = "fit():\t"
        samples_count = len(X)
        self.logger.log_message(signature, "sample_count=", samples_count)
        
        beg = 0
        batch_beg = 0
        batch_X_list = []
        batch_count = 0
        while beg < samples_count:
            end = min(beg + config.MODEL.BATCH_SIZE, samples_count)
            self.logger.log_message("word2vec[{:d}:{:d}]".format(beg, end))
            batch_X_list.append(self.strs_to_numpy(X[beg: end]))
            batch_count += 1
            beg = end
            
            if batch_count >= config.MODEL.CPU_BATCH_TIMES:
                batch_y = np.array(out[batch_beg: end])
                batch_X = np.concatenate(batch_X_list)
                self.logger.log_message("fitting[{:d}:{:d}]".format(batch_beg, end))
                self.partial_fit(batch_X, batch_y)
                batch_beg = end
                batch_X_list.clear()
                batch_count = 0

        if batch_count > 0 :
            batch_y = np.array(out[batch_beg: samples_count])
            batch_X = np.concatenate(batch_X_list)
            self.logger.log_message("fitting[{:d}:{:d}]".format(batch_beg, samples_count))
            self.partial_fit(batch_X, batch_y)

        return self.save()

    def predict(self, samples):
        ret = []
        samples_count = len(samples)
        beg = 0
        while beg < samples_count:
            end = min(beg + config.MODEL.BATCH_SIZE, samples_count)
            X = self.strs_to_numpy(samples[beg: end])
            if self.dim_reducer is not None:
                X = self.dim_reducer.transform(X)
            Y = self.classifier.predict(X)
            for i in range(len(Y)):
                ret.append(int(Y[i]))
            beg = end
        return ret

    def save(self, file_dir=None):
        """
        @param file_dir: 将向该目录下写入保存文件, 会先清空该目录下所有文件和子目录
        """
        signature = "save():\t"
        if file_dir is None:
            file_dir = os.path.join(config.PATH.CKPOINT, self.logger.get_fs_legal_time_stampe())
        try:
            shutil.rmtree(file_dir)
            self.logger.log_message(signature, "clean up dir [", file_dir, ']')
        except FileNotFoundError:
            self.logger.log_message(signature, "makedirs: [", file_dir, ']')
        os.makedirs(file_dir)
        
        if self.dim_reducer is not None:
            joblib.dump(self.dim_reducer, os.path.join(file_dir, "dim_reducer"))
        joblib.dump(self.classifier, os.path.join(file_dir, "classifier"))

        self.logger.log_message("save in [", file_dir, ']')

        return file_dir
    
    def load(self, file_dir):
        dim_reducer_path = os.path.join(file_dir, "dim_reducer")
        if os.path.exists(dim_reducer_path):
            self.dim_reducer = joblib.load(dim_reducer_path)
        else:
            self.dim_reducer = None
        self.classifier = joblib.load(os.path.join(file_dir, "classifier"))
        self.logger.log_message("load from [", file_dir, ']')
        return self


        



def test(model, samples, labels):
    correct_count = 0
    error_count = 0
    beg = 0
    samples_count = len(samples)
    while beg < samples_count:
        end = min(beg + config.MODEL.BATCH_SIZE * config.MODEL.CPU_BATCH_TIMES, samples_count)
        print("checking[{:d}:{:d}]".format(beg, end))
        answer = model.predict(samples[beg: end])
        batch_correct_count = 0
        batch_error_count = 0
        for i in range(len(answer)):
            if labels[beg + i] == answer[i]:
                batch_correct_count += 1
            else:
                batch_error_count += 1
        print("correct/error={:d}/{:d}".format(batch_correct_count, batch_error_count))
        correct_count += batch_correct_count
        error_count += batch_error_count
        beg = end
    print("correct_count=", correct_count)
    print("error_count=", error_count)
    total_count = correct_count + error_count
    print("total_count=", total_count)
    correct_rate = correct_count / total_count
    print("correct_rate=", correct_rate)
    return correct_rate

if __name__ == "__main__":
    samples = []
    labels = []
    count = 0
    for info in loader.walk_csv(config.PATH.DATA_TRAIN_CSV):
        samples.append(info.content)
        labels.append(int(info.positive))
        count += 1
        if count >= 2000:
            break

    model = Model(None)
    ckpoint = model.fit(samples, labels)

    test(model, samples, labels)
    
    model.load(ckpoint)

    test(model, samples, labels)