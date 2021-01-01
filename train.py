from core.preprocessor import loader
from core.model.model import Model
from core.model.model import test
from core.config import config
import random

def train(classifier=None):
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    for info in loader.walk_csv(config.PATH.DATA_TRAIN_CSV):
        if random.random() < 0.7:
            train_samples.append(info.content)
            train_labels.append(int(info.positive))
        else:
            test_samples.append(info.content)
            test_labels.append(int(info.positive))

    model = Model(classifier)
    ckpoint = model.fit(train_samples, train_labels)
    test(model, test_samples, test_labels)
    return model

if __name__ == "__main__":
    train()