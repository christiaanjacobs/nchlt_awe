import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np
class lstm(nn.Module):      
    def __init__(self, options_dict):
        super(lstm, self).__init__()
        self.input_size = options_dict['n_input']
        self.hidden_size = options_dict['hidden_size']
        self.num_layers = options_dict['num_layers']
        self.bias = options_dict['bias']
        self.batch_first = options_dict['batch_first']
        self.dropout = options_dict['dropout']
        self.bidirectional = options_dict['bidirectional']
        self.batch_size = options_dict['batch_size']
        self.n_classes = options_dict['n_classes']
        self.device = options_dict['device']
        self.embedding_dim = options_dict['ff_n_hiddens']
        self.rnn_type = options_dict['rnn_type']
        
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

                            
        # self.lstms = nn.ModuleList()
        # for i in range(self.num_layers):
        #     input_size = self.input_size if i == 0 else self.hidden_size*self.num_directions
        #     self.lstms.append(nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=1, bias=self.bias, 
        #                          batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional))

        # Encoding
        if self.rnn_type == 'lstm': 
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=self.bias, 
                                batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)
        elif self.rnn_type == 'gru':
            self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=self.bias, 
                                batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)


        self.fc = nn.Linear(self.hidden_size*self.num_directions, self.embedding_dim)

        # Output
        self.fc2 = nn.Linear(self.embedding_dim, self.n_classes)
        
            
    def init_hidden(self,):
        h_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float32, device=self.device)
        c_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float32, device=self.device)
        return (h_0, c_0)
        
    def forward(self, x, lengths):
        
        if self.training:                
            # pack padded sequences
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

            # packed_outputs = [] # will contain all the hidden states of each individual LSTM layer.

            #  provide layer with output of prev layer except for the first layer where the input is the training data 
            # for i in range(self.num_layers):
            #     layer_input = packed if i == 0 else output             
            #     output, (hidden, cell) = self.lstms[i](layer_input, self.init_hidden()) # output: tensor of shape (batch_size, seq_length, hidden_size*num_directions) 
            #     packed_outputs.append(output) 

            # output_lstm, (hidden, cell) = self.lstm(packed, self.init_hidden())
            # encode = F.relu(self.fc(hidden[-1].squeeze(0)))
            if self.rnn_type == 'lstm':
                output_lstm, (hidden, _) = self.lstm(packed) # manually set init cell state at init_hidden() - default zero
            elif self.rnn_type == 'gru':
                output_gru, hidden = self.gru(packed)                        
            
            encode = F.relu(self.fc(hidden[-1].squeeze(0)))
            output = self.fc2(encode)
            # hidden contains the last hidden state = out for each sequence at time step equal to original length            
            # unpack only necessay if you need intermediate states otherwise use hidden of shape (batch, hidden_size)
            
            # unpack, lengths = pad_packed_sequence(output, batch_first=True)
            # unpack contains hidden state at each time step up to max seq length, resulting output[:,-1,:] contains zeros in samples where
            # seq_len < than max seq_len. needs to use lengths when unpack to find correct index in output that contains last state of seq n.
            out = output

        else:
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            if self.rnn_type == 'lstm':
                output_lstm, (hidden, _) = self.lstm(packed) # manually set init cell state at init_hidden() - default zero
            elif self.rnn_type == 'gru':
                output_gru, hidden = self.gru(packed)

            out = self.fc(hidden[-1])

        return out


class encoder_rnn(nn.Module):      
    def __init__(self, options_dict):
        super(encoder_rnn, self).__init__()
        self.input_size = options_dict['n_input']
        self.hidden_size = options_dict['hidden_size']
        self.num_layers = options_dict['num_layers']
        self.bias = options_dict['bias']
        self.batch_first = options_dict['batch_first']
        self.dropout = options_dict['dropout']
        self.bidirectional = options_dict['bidirectional']
        self.batch_size = options_dict['batch_size']
        self.device = options_dict['device']
        self.embedding_dim = options_dict['ff_n_hiddens']
        self.rnn_type = options_dict['rnn_type']
        
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # Encoding
        if self.rnn_type == 'lstm': 
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=self.bias, 
                                batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)
        elif self.rnn_type == 'gru':
            self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=self.bias, 
                                batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)
        
        self.fc = nn.Linear(self.hidden_size*self.num_directions, self.embedding_dim)  
        self.fc2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.fc2 = nn.Linear(self.embedding_dim, 60)
        # self.fc3 = nn.Linear(60, 20)
        # self.fc4 = nn.Linear(20, 1)

        # self.adapt = True
        # if self.adapt == True:
        #     self.fc2 = nn.Linear(self.embedding_dim, 130)

        
    def init_hidden(self,):
        h_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float32, device=self.device)
        c_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float32, device=self.device)
        return (h_0, c_0)

        
    def forward(self, x, lengths):            
        # training mode
        if self.training:
            # enforce batch first 
            if not self.batch_first:
                x = permute(1,0,2)   # change x from shape (seq_len, batch, input_size) to shape (batch_size, seq_len, input_size)
            
            # pack padded sequences
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            
            # rnn 
            if self.rnn_type == 'lstm':
                output_lstm, (hidden, _) = self.lstm(packed) # manually set init cell state at init_hidden() - default zero
            elif self.rnn_type == 'gru':
                output_gru, hidden = self.gru(packed)

            # embedded layer
            out = self.fc(hidden[-1].squeeze())
            # out = self.fc2(out)

            # # squeezing to one and softmax
            # out = torch.relu(self.fc2(out))
            # out = torch.relu(self.fc3(out))
            # out = self.fc4(out)
            

            # if self.adapt == True:
            #     out = self.fc2(out)
        
        # validation mode
        else:
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

            if self.rnn_type == 'lstm':
                output_lstm, (hidden, _) = self.lstm(packed) # adjust batch size when calling init_hidden() when train and val batch sizes not equal
            elif self.rnn_type == 'gru':
                output_rnn, hidden = self.gru(packed)
                
                # embedded layer   
            out = self.fc(hidden[-1])
            
            # if self.adapt == True:
            #     out = self.fc2(out)
        # shape (batch, hidden_dim) e.g (300, 130)
        return out


class siamese_rnn(nn.Module):
    def __init__(self, options_dict):
        super(siamese_rnn, self).__init__()
        self.encoder = encoder_rnn(options_dict)

    def forward(self, x, lengths):
        encoder_rnn_output = self.encoder(x, lengths)
        return encoder_rnn_output
            

class decoder_rnn(nn.Module):
    def __init__(self, options_dict):
        super(decoder_rnn, self).__init__()
        self.input_size = options_dict['n_input']
        self.hidden_size = options_dict['hidden_size']
        self.num_layers = options_dict['num_layers']
        self.bias = options_dict['bias']
        self.batch_first = options_dict['batch_first']
        self.dropout = options_dict['dropout']
        self.bidirectional = options_dict['bidirectional']
        self.batch_size = options_dict['batch_size']
        self.device = options_dict['device']
        self.embedding_dim = options_dict['ff_n_hiddens']
        self.rnn_type = options_dict['rnn_type'] 

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        
        # recurrent layers (decoding)
        if self.rnn_type == 'lstm':                                                                                  # create sequence of length equal to encoder input sequence length
            self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=self.bias, 
                                batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)
        elif self.rnn_type == 'gru':
            self.gru = nn.GRU(self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=self.bias, 
                              batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)
        
        # final decoded layer
        self.fc = nn.Linear(self.hidden_size*self.num_directions, self.input_size)


    
    def init_hidden(self,):
        h_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float32, device=self.device)
        c_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float32, device=self.device)
        return (h_0, c_0)

    
    def forward(self, x, lengths):

        if self.training:
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False) # x contains embedded layer repeated to
            
            if self.rnn_type == 'lstm':                                                                                  # create sequence of length equal to encoder input sequence length
                decoder_rnn_output, (_, _) = self.lstm(packed)
            elif self.rnn_type == 'gru':
                decoder_rnn_output, _ = self.gru(packed)

            unpacked_rnn_output, _ = pad_packed_sequence(decoder_rnn_output, batch_first=True) # (batch, max_seq_length, hidden_size)

            final_output = self.fc(unpacked_rnn_output) # (batch, max_seq_len, 13)

            # create mask that replace values in rows wirh zeros at indices greater that original seq_len
            mask = [torch.ones(lengths[i],final_output.size(-1)) for i in range(final_output.size(0))]
            mask_padded = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).to(self.device)

            # apply mask to final_output
            final_output_masked = torch.mul(final_output, mask_padded)
            # print(final_output_masked)

            # Find way to send output into linear layer without calculating gradients on padded values (remove or ignore, take out data, compute and rewrap)
            ## Take out data from packed object, compute linear calculation, rewrap
            # output_decoded_linear = self.fc(output_decoder_rnn.data)
            # output_packed_wrap = torch.nn.utils.rnn.PackedSequence(output_decoded_linear, output_decoder_rnn.batch_sizes)
            # unpacked_output, _ = pad_packed_sequence(output_packed_wrap, batch_first=True) # (batch, max_seq_length, hidden_size
            # unpadded_sequences = [seq[:lengths[i].int()] for i, seq in enumerate(unpacked_output)]
           
        return final_output_masked


class ae_rnn(nn.Module):
    def __init__(self, options_dict):
        super(ae_rnn, self).__init__()
        self.encoder = encoder_rnn(options_dict)
        self.decoder = decoder_rnn(options_dict)



    def forward(self, x, lengths):

        if self.training:

            # encode padded sequences
            encoded_x = self.encoder(x, lengths) # shape (batch, embed_dim)

            # apply activation on embeded layer (necessary?)
            encoded_x = torch.nn.functional.relu_(encoded_x)
            
            lengths = [int(length.tolist()) for length in lengths]

            # Decoding
            # need to repeat latent embedding as input to the rnn up to original sequence length (lengths are already sorted in decending)
            sequences = [z.unsqueeze(0).expand(lengths[i],-1) for i, z in enumerate(encoded_x)] # [batch, seq_len_i, hidden_dim]
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True) # pad variable length sequences to max seq length

            # decode padded sequences
            decoded_x = self.decoder(sequences_padded, lengths)

            return decoded_x

        # only return encoded embeddings
        else:
            return self.encoder(x, lengths)


class cae_rnn(nn.Module):
    def __init__(self, options_dict):
        super(cae_rnn, self).__init__()
        self.encoder = encoder_rnn(options_dict)
        # self.fc2 = nn.Linear(options_dict['ff_n_hiddens'], 130)
        # options_dict['ff_n_hiddens'] = 130
        self.decoder = decoder_rnn(options_dict)

    def forward(self, x, input_lengths, corr_lengths=None):
        
        if self.training:
            # encode padded sequence, x
            encoded_x = self.encoder(x, input_lengths)
            # F.relu_(encoded_x)
            # encoded_x = self.fc2(encoded_x)
            
            # apply activation on embeded layer (necessary?)
            # encoded_x = torch.nn.functional.relu_(encoded_x)
            
            # Decoding
            # repeat latent embedding as input to the rnn up to corresponding output sequence length (corresponding lengths are not sorted)
            lengths_sorted = [(length,idx)  for (idx,length) in sorted(enumerate(corr_lengths), key=lambda x:x[1], reverse=True)]
            corr_lengths_sorted = [x[0].tolist() for x in lengths_sorted]
            corr_sorting_indices = [x[1] for x in lengths_sorted] # use to rearange corr_lengths to orignal sequence

            encoded_x[range(len(encoded_x))] = encoded_x[corr_sorting_indices]

            corr_sequences = [z.unsqueeze(0).expand(corr_lengths_sorted[i],-1) for i, z in enumerate(encoded_x)] # [batch, seq_len_i, hidden_dim]
            corr_sequences_padded = torch.nn.utils.rnn.pad_sequence(corr_sequences, batch_first=True) # pad variable length sequences to max seq length

            # decode padded sequences
            decoded_x = self.decoder(corr_sequences_padded, corr_lengths_sorted)

            # reorder output in orignal order
            final_decoded_x = torch.zeros_like(decoded_x)
            for i in range(len(decoded_x)):                
                final_decoded_x[corr_sorting_indices[i]] = decoded_x[i]

            return final_decoded_x

        else:
            encoded_x = self.encoder(x, input_lengths)
            # F.relu_(encoded_x)
            # encoded_x = self.fc2(encoded_x)
            return encoded_x


class cae_rnn_language_specific(nn.Module):
    def __init__(self, options_dict):
        super(cae_rnn_language_specific, self).__init__()
        self.encoder = encoder_rnn(options_dict)
        # self.decoder_list = [decoder_rnn(options_dict) for i in range(2)]
        self.decoder1 = decoder_rnn(options_dict)
        self.decoder2 = decoder_rnn(options_dict)
        

    def forward(self, x, input_lengths, language_list=None, corr_lengths=None):
        
        if self.training:
            # encode padded sequence, x
            encoded_x = self.encoder(x, input_lengths)
            
            language_list = np.array(language_list)
            languages = np.unique(language_list) # ["GE", "RU"]

            decoded_x_final = torch.zeros((300, torch.max(corr_lengths), 13), dtype=torch.float32, device='cuda')
            for lang in languages:
                lang_index = [i for i, item in enumerate(language_list) if item==lang] # [2, 3, 4, 7, 9, 12, ...]
                encoded_x_lang = encoded_x[lang_index]
                corr_lengths_lang = corr_lengths[lang_index] # [98, 92, 87, 72, ...]
                # now sort according to corr lang seq length
                sorted_list = [(length,idx)  for (idx,length) in sorted(enumerate(corr_lengths_lang), key=lambda x:x[1], reverse=True)]
                corr_lengths_sorted = [x[0].tolist() for x in sorted_list]
                corr_sorted_indices = [x[1] for x in sorted_list] 
                
                encoded_x_sorted = encoded_x_lang[corr_sorted_indices]
                corr_sequences = [z.unsqueeze(0).expand(corr_lengths_sorted[i],-1) for i, z in enumerate(encoded_x_sorted)]
                corr_sequences_padded = torch.nn.utils.rnn.pad_sequence(corr_sequences, batch_first=True) # pad variable length sequences to max seq length
                # decoded_x = self.decoder_dict[lang](corr_sequences_padded, corr_lengths_sorted)
                if lang == 'RU':
                    decoded_x = self.decoder1(corr_sequences_padded, corr_lengths_sorted)
                else:
                    decoded_x = self.decoder2(corr_sequences_padded, corr_lengths_sorted)
                decoded_x_unsorted = torch.zeros_like(decoded_x)
                decoded_x_unsorted[corr_sorted_indices] = decoded_x
                # print(decoded_x_unsorted.shape)
                # print(decoded_x.size(1))
                for i, item in enumerate(decoded_x_unsorted):
                    decoded_x_final[lang_index[i], :len(item)] = item               
            return decoded_x_final                
        else:
            return self.encoder(x, input_lengths)