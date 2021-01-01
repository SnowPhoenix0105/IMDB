from core.preprocessor import loader
from core.model.model import Model
from core.config import config
import os
from train import train


if __name__ == "__main__":
    model = Model(None)
    model.load(os.path.join(config.PATH.CKPOINT, "2021-01-01_22-08-41"))

    # model = train()

    samples = loader.walk_csv(config.PATH.DATA_TEST_CSV)
    samples_it = iter(samples)


    with open(config.PATH.DATA_RESULT_CSV, 'w', encoding='utf8') as fout:
        with open(config.PATH.DATA_TEMPLEMENT_CSV, 'r', encoding='utf8') as fin:
            sample_batch = []
            line_batch = []
            batch_count = 0
            beg = 0
            end = 0
            for row, line in enumerate(fin):
                if row == 0:
                    fout.write(line.strip())
                    beg += 1
                    continue
                end = row
                sample = next(samples_it)
                sample_batch.append(sample.content)
                line_batch.append(line.strip())
                assert sample.ID == int(line_batch[-1][:-9])
                batch_count += 1
                if batch_count < config.MODEL.BATCH_SIZE * config.MODEL.CPU_BATCH_TIMES:
                    continue
                print("checking[{:d}:{:d}]".format(beg, end))
                beg = end
                label_batch = model.predict(sample_batch)
                for content, label in zip(line_batch, label_batch):
                    fout.write('\n' + content[:-8] + ("positive" if label != 0 else "negative"))
                line_batch.clear()
                sample_batch.clear()
                batch_count = 0
            print("checking[{:d}:{:d}]".format(beg, end))
            if batch_count != 0:
                label_batch = model.predict(sample_batch)
                for content, label in zip(line_batch, label_batch):
                    fout.write('\n' + content[:-8] + ("positive" if label != 0 else "negative"))
            

    os.system("7z a -tzip " + config.PATH.DATA_RESULT_CSV.replace(".csv", '.zip') + " " + config.PATH.DATA_RESULT_CSV)