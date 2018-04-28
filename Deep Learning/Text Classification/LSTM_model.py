import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, 
                 label_size, batch_size, n_layers, dropout_p, 
                 pretrained_embeddings=None, USE_CUDA=False):
        super(LSTMClassifier, self).__init__()
        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = label_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.USE_CUDA = USE_CUDA
        
        self.embedding = nn.Embedding(self.input_size, embedding_size)
        if pretrained_embeddings is not None:
            self.embedding.load_state_dict({'weight': pretrained_embeddings})
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, 
                            bidirectional=True, dropout=dropout_p)
        self.out = nn.Linear(2 * hidden_size, self.output_size)
        self.hidden = self.init_hidden(batch_size)
        
    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size))
        cell = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size))
        if self.USE_CUDA:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)
    
    def forward(self, sentence_batch):
        '''
        sentence_batch: (seq_len, BS)
        output: (BS, output_size)
        '''
        embedded = self.embedding(sentence_batch).cuda()
        # self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embedded, self.hidden)
        output = torch.cat((outputs[-1, :, :self.hidden_size],
                            outputs[ 0, :, self.hidden_size:]), 1)
        output = self.out(output)
        output = F.log_softmax(output, 1)
        return output    
    
def train_step(input_batch, targets, model, optimizer,
          criterion, USE_CUDA, train=True, clip=5.0):
    if train:
        optimizer.zero_grad()
    loss = 0
    batch_size = input_batch.size()[1]
    model.hidden = model.init_hidden(batch_size)
    output = model(input_batch)
    loss = nn.NLLLoss()(output, targets)
    if train:
        loss.backward()
        torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, 
                                             model.parameters()), clip)
        optimizer.step()
    return loss.data[0]    

def evaluate(input_batch, model, USE_CUDA):
    batch_size = input_batch.size()[1]
    model.hidden = model.init_hidden(batch_size)
    output = model(input_batch)
    return output