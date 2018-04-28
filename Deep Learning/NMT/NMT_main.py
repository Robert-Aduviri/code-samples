import random
import time
import random
import torch.nn as nn
from torch import optim

from NMT_preprocessing import prepare_data
from NMT_preprocessing import construct_vectors
from NMT_preprocessing import random_batch
from NMT_model import EncoderRNN
from NMT_model import AttnDecoderRNN
from NMT_model import train
from NMT_utils import as_minutes
from NMT_utils import time_since

class NMT:
    def __init__(self, *args_dict, **kwargs):
        self.MAX_LENGTH = 15
        self.USE_CUDA = True

        self.attn_model = 'general'
        self.embedding_size = 300
        self.hidden_size = 512
        self.n_layers = 2
        self.encoder_dropout_p = 0.0
        self.decoder_dropout_p = 0.5

        self.n_epochs = 5000
        self.plot_every = 10
        self.print_every = 10
        self.validate_every = 50

        self.batch_size = 128

        for dictionary in args_dict:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def train_val_split(self, pairs):
        random.seed(42)
        random.shuffle(pairs, random.random)
        return pairs[:-1000], pairs[-1000:], pairs

    def get_model(self, input_lang, output_lang, eng_vectors, spa_vectors):
        encoder = EncoderRNN(input_lang.n_words, self.embedding_size, 
                             self.hidden_size, self.batch_size, 
                             self.n_layers, self.encoder_dropout_p, 
                             eng_vectors, self.USE_CUDA)
        decoder = AttnDecoderRNN(output_lang.n_words, self.embedding_size, 
                             self.hidden_size, self.attn_model, 
                             self.n_layers * 2, self.decoder_dropout_p, 
                             spa_vectors, self.USE_CUDA)
        if self.USE_CUDA:
            encoder.cuda()
            decoder.cuda()
        learning_rate = 0.001
        encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                                        encoder.parameters()), lr=learning_rate)
        decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                                        decoder.parameters()), lr=learning_rate)
        criterion = nn.NLLLoss()

        return encoder, decoder, encoder_optimizer, decoder_optimizer, criterion

    def train_model(self, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    criterion, input_lang, output_lang, train_data, val_data):
        train_losses = []
        validation_losses = []
        print_loss_total = 0
        plot_loss_total = 0
        start = time.time()
        for epoch in range(1, self.n_epochs + 1):  
            input_batches, input_lengths, target_batches, target_lengths = random_batch(
                        self.batch_size, input_lang, output_lang, 
                        train_data, self.USE_CUDA)
            
            loss = train(input_batches, input_lengths, target_batches, target_lengths,\
                        encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 
                        self.USE_CUDA, train=True)
            
            print_loss_total += loss
            plot_loss_total += loss
            
            if epoch == 0: continue
                
            if epoch % self.print_every == 0:
                print_loss_avg = print_loss_total / self.print_every
                print_loss_total = 0
                print(f'{time_since(start, epoch / self.n_epochs)} ({epoch} {epoch / self.n_epochs * 100:.2f}%) train_loss: {print_loss_avg:.4f}', end=' ')
            
            if epoch % self.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.plot_every
                train_losses.append(plot_loss_avg)
                plot_loss_total = 0
                
            if epoch % self.validate_every == 0:
                input_batches, input_lengths, target_batches, target_lengths = random_batch(
                        self.batch_size, input_lang, output_lang, val_data)
            
                eval_loss = train(input_batches, input_lengths, target_batches, target_lengths,
                            encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 
                            self.USE_CUDA, train=False)
                
                validation_losses.append(eval_loss)
                print(f'- val_loss: {eval_loss:.4f}', end='')
            
            if epoch % self.print_every == 0:
                print()

        return encoder, decoder, train_losses, validation_losses

    def main(self, *args_dict, **kwargs):
        corpus = 'parallel-corpora/eng-es/eng-spa.txt'
        in_lang, out_lang = 'eng', 'spa'
        input_lang, output_lang, pairs = prepare_data(corpus, in_lang, out_lang, 
                                                        self.MAX_LENGTH)        
        train_data, val_data, pairs = self.train_val_split(pairs)
        eng_vectors, spa_vectors = construct_vectors(pairs)

        return input_lang, output_lang, eng_vectors, spa_vectors

        encoder, decoder, encoder_optimizer, decoder_optimizer, criterion = self.get_model(
                            input_lang, output_lang, eng_vectors, spa_vectors)

        encoder, decoder, train_losses, validation_losses = self.train_model(
            encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
            input_lang, output_lang, train_data, val_data)

        return encoder, decoder, input_lang, output_lang, train_losses, validation_losses
                        

        