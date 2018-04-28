import random
import numpy as np
from tqdm import tqdm, tqdm_notebook
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from NMT_preprocessing import variable_from_sentence
from NMT_preprocessing import SOS_token
from NMT_preprocessing import EOS_token

############## EVALUATE MODEL ##########################

class Beam():
    def __init__(self, decoder_input, decoder_context, decoder_hidden, decoder_cell,
                    decoded_words=[], decoder_attentions=[], sequence_log_probs=[]):
        self.decoded_words = decoded_words
        self.decoder_attentions = decoder_attentions
        self.sequence_log_probs = sequence_log_probs
        self.decoder_input = decoder_input
        self.decoder_context = decoder_context
        self.decoder_hidden = decoder_hidden
        self.decoder_cell = decoder_cell

class Evaluator():
    def __init__(self, encoder, decoder, input_lang, output_lang, max_length, USE_CUDA):
        self.encoder = encoder
        self.decoder = decoder
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length
        self.USE_CUDA = USE_CUDA

    def evaluate(self, sentence, k_beams):
        input_variable = variable_from_sentence(self.input_lang, sentence, self.USE_CUDA)
        input_length = input_variable.shape[0]
        
        encoder_hidden = self.encoder.init_hidden(1)
        encoder_cell = self.encoder.init_cell(1)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(input_variable, input_length, encoder_hidden, encoder_cell)
        
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        
        if self.USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
            
        decoded_words = []
        decoder_attentions = torch.zeros(self.max_length, self.max_length)

        beams = [Beam(decoder_input, decoder_context, decoder_hidden, decoder_cell)]
        top_beams = []
        
        # Use decoder output as inputs
        for di in range(self.max_length):      
            new_beams = []
            for beam in beams:
                decoder_output, decoder_context, (decoder_hidden, decoder_cell), decoder_attention = self.decoder(
                    beam.decoder_input, beam.decoder_context, beam.decoder_hidden, beam.decoder_cell, encoder_outputs)     

                # Beam search, take the top k with highest probability
                topv, topi = decoder_output.data.topk(k_beams)

                for ni, vi in zip(topi[0], topv[0]):
                    new_beam = Beam(None, decoder_context, decoder_hidden, decoder_cell, 
                                    beam.decoded_words[:], beam.decoder_attentions[:], beam.sequence_log_probs[:])
                    new_beam.decoder_attentions.append(decoder_attention.squeeze().cpu().data)
                    new_beam.sequence_log_probs.append(vi)

                    if ni == EOS_token: 
                        new_beam.decoded_words.append('<EOS>')
                        top_beams.append(new_beam)

                    else:
                        new_beam.decoded_words.append(self.output_lang.index2word[ni])                        
                    
                        decoder_input = Variable(torch.LongTensor([[ni]]))
                        if self.USE_CUDA: decoder_input = decoder_input.cuda()
                        new_beam.decoder_input = decoder_input                        
                        new_beams.append(new_beam)                    
            
            new_beams = {beam: np.mean(beam.sequence_log_probs) for beam in new_beams}
            beams = sorted(new_beams, key=new_beams.get, reverse=True)[:k_beams]

            if len(beams) == 0:
                break
        
        top_beams = {beam: np.mean(beam.sequence_log_probs) for beam in top_beams}

        # for beam in top_beams:
        #     print(beam.decoded_words, top_beams[beam])

        top_beams = sorted(top_beams, key=top_beams.get, reverse=True)[:k_beams]        

        decoded_words = top_beams[0].decoded_words
        for di, decoder_attention in enumerate(top_beams[0].decoder_attentions):
            decoder_attentions[di,:decoder_attention.size(0)] += decoder_attention

        return decoded_words, decoder_attentions[:len(top_beams[0].decoder_attentions)+1, :len(encoder_outputs)], top_beams

    def evaluate_sentence(self, sentence, k_beams=3):        
        output_words, decoder_attn, beams = self.evaluate(sentence, k_beams)
        output_sentence = ' '.join(output_words)
        
        print('>', sentence)
        print('<', output_sentence)
        print('')

    def evaluate_randomly(self, pairs, k_beams=3):
        pair = random.choice(pairs)
        print(pair)
        
        output_words, decoder_attn, beams = self.evaluate(pair[0], k_beams)
        output_sentence = ' '.join(output_words)
        
        print('>', pair[0])
        print('=', pair[1])
        print('<', output_sentence)
        print('')

    def show_attention(self, input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
        plt.close()

    def evaluate_and_show_attention(self, input_sentence, k_beams=3):
        output_words, attentions, beams = self.evaluate(input_sentence, k_beams)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        self.show_attention(input_sentence, output_words, attentions)

    def evaluate_randomly_and_show_attention(self, pairs, k_beams=3):
        pair = random.choice(pairs)
        print(pair)        
        self.evaluate_and_show_attention(pair[0], k_beams)

    def get_candidates_and_references(self, pairs, k_beams=3):
        candidates = [self.evaluate(pair[0], k_beams)[0] for pair in tqdm_notebook(pairs)]
        candidates = [' '.join(candidate[:-1]) for candidate in candidates]
        references = [pair[1] for pair in pairs]
        return candidates, references




