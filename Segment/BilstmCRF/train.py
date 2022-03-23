import tensorflow as tf
import numpy as np
import os, argparse, time, random
from util import str2bool,get_logger
from model import BiLSTM_CRF
from data import read_dictionary, random_embedding,read_corpus,tag2label

## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='ProcessData', help='train data source')
parser.add_argument('--test_data', type=str, default='ProcessData', help='test data source')
parser.add_argument('--batch_size', type=int, default=32, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=30, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=100, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1516849095', help='model for test and demo')
args = parser.parse_args()


timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
# 输出文件地址
output_path = os.path.join('', args.train_data + "_save", "1558343713")
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
result_path = os.path.join(output_path, "results")
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
get_logger(log_path).info(str(args))

def getDicEmbed():
    word2id = read_dictionary(os.path.join('', args.train_data, 'word2id.pkl'))
    if args.pretrain_embedding == 'random':
        embeddings = random_embedding(word2id, args.embedding_dim)
    else:
        embedding_path = 'pretrain_embedding.npy'
        embeddings = np.array(np.load(embedding_path), dtype='float32')

    return word2id, embeddings




def getTrainData(filename):
    return read_corpus(filename)

def Train(trainfile):
    word2id, embeddings = getDicEmbed()
    traindata = getTrainData(trainfile)

    model = BiLSTM_CRF(batch_size=args.batch_size,
                       epoch_num=args.epoch,
                       hidden_dim=args.hidden_dim,
                       embeddings=embeddings,
                       dropout_keep=args.dropout,
                       optimizer=args.optimizer,
                       lr=args.lr,
                       clip_grad=args.clip,
                       tag2label=tag2label,
                       vocab=word2id,
                       shuffle=args.shuffle,
                       model_path=ckpt_prefix,
                       summary_path=summary_path,
                       log_path=log_path,
                       result_path=result_path,
                       CRF=args.CRF,
                       update_embedding=args.update_embedding)
    model.build_graph()

    dev_data = traindata[:5000]
    dev_size = len(dev_data)
    train_data = traindata[5000:]
    train_size = len(train_data)
    print("train data: {0}\n dev data: {1}".format(train_size, dev_size))
    model.train(traindata,dev_data)

def Test(testfile):
    word2id, embeddings = getDicEmbed()
    testdata = getTrainData(testfile)
    ckpt_file = tf.train.latest_checkpoint(model_path)
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embedding)
    model.build_graph()

    model.test(testdata)


def predict_random(demo_sent):
    word2id, embeddings = getDicEmbed()
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim,
                       embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embedding)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        demo_sent = list(demo_sent.strip())
        demo_data = [(demo_sent, ['M'] * len(demo_sent))]
        tag = model.demo_one(sess, demo_data)
        sess.close()
    res = segment(sent,tag)
    print(res)



def segment(sent,tag):
    res = ''
    for i in range(len(sent)):
        if tag[i] == 'S' or tag[i] == 'E':
            res+=sent[i]+" "
        else:
            res+=sent[i]
    return res.split()

if __name__ == '__main__':
    # trainfile =os.path.join(args.train_data,'train.utf8')
    # Train(trainfile)
    # testfile = os.path.join(args.test_data,'test.utf8')
    # Test(testfile)
    sent = "不要有列表的嵌套的形式"
    predict_random(sent)