import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from NMT_preprocessing import SOS_token

############## CREATE MODEL ##########################

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, batch_size, n_layers=1, dropout_p=0.0, lang=None, USE_CUDA=False):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.batch_size = batch_size
        self.USE_CUDA = USE_CUDA
        
        # (size of dictionary of embeddings, size of embedding vector)
        self.embedding = nn.Embedding(input_size, embedding_size)
        if(lang):
            self.embedding.weight.data.copy_(lang.vocab.vectors)
            self.embedding.weight.requires_grad = False
        # (input features, hidden state features, number of layers)
#         self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=True)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, bidirectional=True, dropout=dropout_p)
        
    def forward(self, word_inputs, input_lengths, hidden=None, cell=None):
        '''
        word_inputs: (seq_len, BS)
        hidden: (n_layers, BS, hidden)
        cell: (n_layers, BS, hidden)
        < output: (seq_len, BS, hidden)
        '''
        # This is run over all the input sequence
        
        seq_len = len(word_inputs)
        # word_inputs: (seq_len, batch_size), values in [0..input_size)
        embedded = self.embedding(word_inputs)#.view(seq_len, 1, -1) 
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        
        # embedded: (seq_len, batch_size, embedding_size)
        # hidden: (2 * num_layers, batch_size, hidden_size)
        # cell: (2 * num_layers, batch_size, hidden_size)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output) # unpack (back to padded)

        # outputs: (seq_len, batch_size, 2 * hidden_size)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        # outputs: (seq_len, batch_size, hidden_size)
        # hidden: same
        return outputs, (hidden, cell)
    
    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        hidden = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size))
        if self.USE_CUDA: hidden = hidden.cuda()
        return hidden
    
    def init_cell(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        cell = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size))
        if self.USE_CUDA: cell = cell.cuda()
        return cell

class LuongGlobalAttn(nn.Module):
    def __init__(self, method, hidden_size, USE_CUDA=False):
        super(LuongGlobalAttn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.USE_CUDA = USE_CUDA
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))
            
    def forward(self, hidden, encoder_outputs):
        '''
        hidden: (BS, hidden_size)
        encoder_outputs(seq_len, BS, encoder_hidden_size)
        '''
        # encoder_outputs: (seq_len, batch_size, encoder_hidden_size)
        seq_len = len(encoder_outputs)
        batch_size = encoder_outputs.shape[1]
        
        # Calculate attention energies for each encoder output
        # attn_energies: (seq_len, batch_size)
        # hidden: (batch_size, hidden_size)
        attn_energies = Variable(torch.zeros(seq_len, batch_size))
        if self.USE_CUDA: attn_energies = attn_energies.cuda()
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        
        # Normalize energies [0-1] and resize to (batch_size, x=1, seq_len)
        return F.softmax(attn_energies, 0).transpose(0, 1).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        # hidden: (batch_size, hidden_size)
        # encoder_output: (batch_size, encoder_hidden_size)
        
        # hidden sizes must match, batch_size = 1 only
        if self.method == 'dot': 
            # batch element-wise dot product
            energy = torch.bmm(hidden.unsqueeze(1), 
                        encoder_output.unsqueeze(2)).squeeze().squeeze()
            # energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            # batch element-wise dot product
            energy = torch.bmm(hidden.unsqueeze(1), 
                        encoder_output.unsqueeze(2)).squeeze().squeeze()
            # energy = hidden.dot(energy)
            return energy
        
        # TODO: test / modify method to support batch size > 1
        elif self.method == 'concat': 
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy  
        
        # energy: (hidden_size)

class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, attn_method,
                    n_layers=1, dropout_p=0.1, lang=None, USE_CUDA=False):
        super(AttnDecoderRNN, self).__init__()
        
        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.USE_CUDA = USE_CUDA
        
        # (size of dictionary of embeddings, size of embedding vector)
        self.embedding = nn.Embedding(output_size, embedding_size)
        if(lang):
            self.embedding.weight.data.copy_(lang.vocab.vectors)
            self.embedding.weight.requires_grad = False
        # (input features: hidden_size + encoder_hidden_size, hidden state features, number of layers)
#         self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size, n_layers, dropout=dropout_p)
        # (input_features: embedding_size + encoder_hidden_size, output_features)
        self.out = nn.Linear(hidden_size * 2, output_size)
        
        self.attn = LuongGlobalAttn(attn_method, hidden_size, USE_CUDA)
        
    def forward(self, word_input, last_context, last_hidden, last_cell, encoder_outputs):
        '''
        word_input: (seq_len, BS)
        last_context: (BS, encoder_hidden_size)
        last_hidden: (n_layers, BS, hidden_size)
        last_cell: (n_layers, BS, hidden_size)
        encoder_outputs: (seq_len, BS, encoder_hidden)
        < output: (BS, output_size)
        < attn_weights: (BS, 1, seq_len)
        '''
        # This is run one step at a time
        
        # Get the embedding of the current input word (last output word)
        # word_input: (seq_len=1, batch_size), values in [0..output_size)
        word_embedded = self.embedding(word_input) #.view(1, 1, -1)
        # word_embedded: (seq_len=1, batch_size, embedding_size)
        
        # Combine embedded input word and last context, run through RNN
        # last_context: (batch_size, encoder_hidden_size)
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        # rnn_input: (seq_len=1, batch_size, embedding_size + encoder_hidden_size)
        # last_hidden: (num_layers, batch_size, hidden_size)
        rnn_output, (hidden, cell) = self.lstm(rnn_input, (last_hidden, last_cell))
        # rnn_output: (seq_len=1, batch_size, hidden_size)
        # hidden: same
        
        # Calculate attention and apply to encoder outputs
        # encoder_outputs: (seq_len, batch_size, encoder_hidden_size)
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        
        # Check softmax:
        # print('attn_weights sum: ', torch.sum(attn_weights.squeeze(), 1))
        
        # attn_weights: (batch_size, x=1, seq_len)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context: (batch_size, x=1, encoder_hidden_size)
        
        # Final output layer using hidden state and context vector
        rnn_output = rnn_output.squeeze(0)
        # rnn_output: (batch_size, hidden_size)
        context = context.squeeze(1)
        # context: (batch_size, encoder_hidden_size)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), 1)
        # output: (batch_size, output_size)
        
        # Check softmax (not log_softmax):
        # print('output sum: ', torch.sum(output.squeeze(), 1))
        
        # Also return attention weights for visualization
        return output, context, (hidden, cell), attn_weights

############## TRAIN MODEL ##########################

def train(input_batches, input_lengths, target_batches, target_lengths, \
          encoder, decoder, encoder_optimizer, decoder_optimizer, \
          criterion, USE_CUDA=False, train=True, teacher_forcing_ratio=0.5,
          clip=5.0):
    
    if train:
        if encoder_optimizer:
            encoder_optimizer.zero_grad()
        if decoder_optimizer:
            decoder_optimizer.zero_grad()
        
    loss = 0    
    batch_size = input_batches.size()[1]
    
    encoder_hidden = encoder.init_hidden()
    encoder_cell = encoder.init_cell()

    if USE_CUDA:
        input_batches = input_batches.cuda()
        encoder_hidden = encoder_hidden.cuda()
        encoder_cell = encoder_cell.cuda()
        target_batches = target_batches.cuda()
    
    encoder_outputs, (encoder_hidden, encoder_cell) = encoder(input_batches, input_lengths, encoder_hidden, encoder_cell)
    
    decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))   
    # decoder_hidden = encoder_hidden[:encoder.n_layers]
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell
    
    # set the start of the sentences of the batch
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))

    # store the decoder outputs to estimate the loss
    all_decoder_outputs = Variable(torch.zeros(target_batches.size()[0], 
                                        batch_size, decoder.output_size))

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
        decoder_context = decoder_context.cuda()  
    
    if train:
        use_teacher_forcing = random.random() < teacher_forcing_ratio
    else:
        use_teacher_forcing = False
    
    if use_teacher_forcing:        
        # Use targets as inputs
        for di in range(target_batches.shape[0]):
            decoder_output, decoder_context, (decoder_hidden, decoder_cell), decoder_attention = decoder(
                decoder_input.unsqueeze(0), decoder_context, decoder_hidden, decoder_cell, encoder_outputs)
            
            all_decoder_outputs[di] = decoder_output
            decoder_input = target_batches[di]
    else:        
        # Use decoder output as inputs
        for di in range(target_batches.shape[0]):            
            decoder_output, decoder_context, (decoder_hidden, decoder_cell), decoder_attention = decoder(
                decoder_input.unsqueeze(0), decoder_context, decoder_hidden, decoder_cell, encoder_outputs) 
            
            all_decoder_outputs[di] = decoder_output
            
            # Greedy approach, take the word with highest probability
            topv, topi = decoder_output.data.topk(1)            
            decoder_input = Variable(torch.LongTensor(topi.cpu()).squeeze())
            if USE_CUDA: decoder_input = decoder_input.cuda()
                            
    log_probs = F.log_softmax(all_decoder_outputs.view(-1, decoder.output_size), dim=1)
    loss = nn.NLLLoss()(log_probs, target_batches.contiguous().view(-1))          
    
    if train:
        loss.backward()
        if encoder_optimizer:
            torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
            encoder_optimizer.step()
        if decoder_optimizer:
            torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
            decoder_optimizer.step()
    
    return loss.data[0] 
