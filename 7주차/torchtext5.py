import json
import torch
from torchtext.datasets import text_classification
from torch.utils.data import DataLoader
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
import torch
import torchtext
from torchtext.datasets import text_classification
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}
word_set = []
text = []
data = []
labels = []
content_word_set = []
content_text = []
vocab = {}

def tokenizer(data_path):
    with open(data_path, 'r', encoding='UTF8') as f:
        json_data = json.load(f)

        # 대화 하나마다의 반복
        for i in json_data:
            json_txt1 = content_text.append(i['talk']['content']['HS01'])
            json_txt2 = content_text.append(i['talk']['content']['SS01'])
            json_txt3 = content_text.append(i['talk']['content']['HS02'])
            json_txt4 = content_text.append(i['talk']['content']['SS02'])
            json_txt5 = content_text.append(i['talk']['content']['HS03'])
            json_txt6 = content_text.append(i['talk']['content']['SS03'])
            json_emotion = str(i['profile']['emotion']['type'])[1:]  # str 처리 되나

            for V in content_text:
                for n in V.split():
                    content_word_set.append(n)

            for i in content_text:
                if i != '':
                    text.append(i)

            print('content_text: ', content_text, '\n')
            print('text: ', text, '\n')
            del content_text[:]


            for i in content_word_set:
                if i != '':
                    word_set.append(i)

            vocab = {word: i + 2 for i, word in enumerate(word_set)}  # 단어 집합의 각 단어에 고유한 정수 맵핑.
            vocab['<unk>'] = 0
            vocab['<pad>'] = 1

            print("vocba: ", vocab, '\n')
            #print("lenwordset: ", len(content_word_set), '\n')
            print("cwordset: ", content_word_set, '\n')

            token_ids = list(filter(lambda x: x, [vocab[token] for token in content_word_set]))
            print("token_ids: ", token_ids, '\n')
            tokens = torch.tensor(token_ids)
            print("token: ", tokens, '\n')

            data.append((json_emotion, tokens))     # emotion으로 수정
            labels.append(json_emotion)

            del content_word_set[0:]

            print("wordset: ", word_set, '\n')

            print("text: ", text, '\n')

            print("data[]: ", data, '\n')
            print("labels[]: ", labels, '\n')

            train_data, train_labels = data, set(labels)

        return TextClassificationDataset(vocab, train_data, train_labels)

#print("test word_set: ", word_set, '\n')

class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, data, labels):
        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab


train_json_path = 'train.json'
test_json_path = 'test.json'

# test_dataset -> test.json 의 vocab을 return
train_json = tokenizer(train_json_path)

BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


VOCAB_SIZE = len(train_json.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_json.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

print("train_getlabels: ", train_json.get_labels(), '\n')
print("train_json", train_json, '\n')
print("vocabsize: ", VOCAB_SIZE, '\n')
print("model: ", model, '\n')
#print("model_forward: ", , '\n')


def predict(text, model, vocab, ngrams):
    # tokenizer = get_tokenizer("basic_english")
    tokenizer = text.split()
    print('tokenizer', tokenizer, '\n')
    with torch.no_grad():
        text = torch.tensor([vocab[token] for token in ngrams_iterator(tokenizer, ngrams)])
        #print("word_set: ", word_set, '\n')     # 테스트 위해 사용
        #print("text: ", text, "\n")     # 테스트 위해 사용
        output = model(text, torch.tensor([0]))
        print("output: ", output, '\n')
        return output.argmax(1).item() + 1


def generate_batch(batch):
    #print('batch: ', batch)
    label = torch.tensor([int(entry[0]) for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum은 dim 차원의 요소들의 누적 합계를 반환합니다.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(sub_train_):
    # Train the model
    # 모델을 학습합니다
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        print("확인text: ", text)
        print("확인off: ", offsets)
        print("확인cls: ", cls)
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        print("확인output: ", output)
        print("확인cls: ", cls)
        print("확인crit: ", criterion(output, cls))
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # 학습률을 조절합니다
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)


def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
print('criterion: ', criterion, '\n')
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_json) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_json, [train_len, len(train_json) - train_len])

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    #    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

ex_text_str = "기분이 너무 건강"
vocab = train_json.get_vocab()
model = model.to("cpu")


print(predict(ex_text_str, model, vocab, 1))     # 테스트 위해 사용
print('Checking the results of test dataset...')
#test_loss, test_acc = test(train_json)
# print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
print("This is a %s news" % ag_news_label[predict(ex_text_str, model, vocab, 1)])
