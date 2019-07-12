from PKUNER.MakeMode import sent2features,sent2labels
from tqdm import tqdm
import pycrfsuite
import pickle

def train(filename,   l1=1.0,   l2=1e-3,    max_iter=200, mode_path ='PKU.crfsuite' ):
    with open(filename, 'rb') as rp:
        train_set = pickle.load(rp)

    trainer = pycrfsuite.Trainer(verbose=False)

    for item in tqdm(train_set):
        trainer.append(sent2features(item),sent2labels(item))

    trainer.set_params({
        'c1': l1,  # coefficient for L1 penalty
        'c2': l2,  # coefficient for L2 penalty
        'max_iterations': max_iter,

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True,
        'feature.minfreq':3
    })
    print("traning mode.........")
    trainer.train(mode_path)
    print("done!")


if __name__ == '__main__':
    train_file = 'train.pkl'
    train(filename=train_file)