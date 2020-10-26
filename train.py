import config
import trainer_step
import tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def main():
    cfg = config.Config(filenamequeue="drive/My Drive/LayoutNet/dataset/layout_1205.tfrecords")
    t = trainer_step.Trainer(cfg)
    t.fit()


if __name__ == "__main__":
    main()
