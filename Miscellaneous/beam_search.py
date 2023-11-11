
import pickle
import random
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
from torch.nn import Module, Linear, Softmax, ReLU, LayerNorm, ModuleList, Dropout, Embedding, CrossEntropyLoss
from torch.optim import Adam

class PositionalEncodingLayer(Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, sequence_length, embedding_dim)

        This function should create the positional encoding matrix
        and return the sum of X and the encoding matrix.

        The positional encoding matrix is defined as follow:

        P_(pos, 2i) = sin(pos / (10000 ^ (2i / d)))
        P_(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d)))

        The output will have shape (batch_size, sequence_length, embedding_dim)
        """
        # TODO: Implement positional encoding
        # raise NotImplementedError()
        # d_model = self.embedding_dim
        # max_positions = X.shape[1]
                
        encoding = torch.zeros(X.shape[1], self.embedding_dim)
        pos = torch.arange(0, X.shape[1]).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / self.embedding_dim))
        div_term = 1/(10000 ** (torch.arange(0, self.embedding_dim, 2)/self.embedding_dim))

        encoding[:, 0::2] = torch.sin(pos * div_term)
        encoding[:, 1::2] = torch.cos(pos * div_term)

        encoding = encoding.unsqueeze(0)
        
        # print("encoding:", encoding.shape, "X:", X.shape, "div:", div_term.shape)        

        emb = encoding[:, :X.size(1), :]

        return emb + X

class SelfAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.linear_Q = Linear(in_dim, out_dim)
        self.linear_K = Linear(in_dim, out_dim)
        self.linear_V = Linear(in_dim, out_dim)

        self.softmax = Softmax(-1)

        self.in_dim = in_dim
        self.out_dim = out_dim


    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query_X, key_X and value_X have shape (batch_size, sequence_length, in_dim). The sequence length
        may be different for query_X and key_X but must be the same for key_X and value_X.

        This function should return two things:
            - The output of the self-attention, which will have shape (batch_size, sequence_length, out_dim)
            - The attention weights, which will have shape (batch_size, query_sequence_length, key_sequence_length)

        If a mask is passed as input, you should mask the input to the softmax, using `float(-1e32)` instead of -infinity.
        The mask will be a tensor with 1's and 0's, where 0's represent entries that should be masked (set to -1e32).

        Hint: The following functions may be useful
            - torch.bmm (https://pytorch.org/docs/stable/generated/torch.bmm.html)
            - torch.Tensor.masked_fill (https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
        """
        # TODO: Implement the self-attention layer
        # raise NotImplementedError()
        
        q = self.linear_Q(query_X)
        k = self.linear_K(key_X)
        v = self.linear_V(value_X)

        batch_size, seq_len, d_model = k.size()

        attention_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(d_model))
        # print("attention_scores:", attention_scores.size(), q.size(), k.transpose(-2, -1).size())
        # print("attention mask:", mask.size(), attention_scores.size())

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e32)

        attention_probs = self.softmax(attention_scores)
        attention_output = torch.bmm(attention_probs, v)

        # attention_output = self.linear(attention_output)

        return attention_output, attention_probs

class MultiHeadedAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention_heads = ModuleList([SelfAttentionLayer(in_dim, out_dim) for _ in range(n_heads)])

        self.linear = Linear(n_heads * out_dim, out_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function calls the self-attention layer and returns the output of the multi-headed attention
        and the attention weights of each attention head.

        The attention_weights matrix has dimensions (batch_size, heads, query_sequence_length, key_sequence_length)
        """

        outputs, attention_weights = [], []

        for attention_head in self.attention_heads:
            out, attention = attention_head(query_X, key_X, value_X, mask)
            outputs.append(out)
            attention_weights.append(attention)

        outputs = torch.cat(outputs, dim=-1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return self.linear(outputs), attention_weights

class EncoderBlock(Module):

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def forward(self, X, mask=None):
        """
        Implementation of an encoder block. Both the input and output
        have shape (batch_size, source_sequence_length, embedding_dim).

        The mask is passed to the multi-headed self-attention layer,
        and is usually used for the padding in the encoder.
        """
        att_out, _ = self.attention(X, X, X, mask)

        residual = X + self.dropout1(att_out)

        X = self.norm1(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)

        residual = X + self.dropout2(temp)

        return self.norm2(residual)

class Encoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([EncoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])
        self.vocab_size = vocab_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transformer encoder. The input has dimensions (batch_size, sequence_length)
        and the output has dimensions (batch_size, sequence_length, embedding_dim).

        The encoder returns its output and the location of the padding, which will be
        used by the decoder.
        """
        # TODO: Implement the encoder (you should re-use the EncoderBlock class that we provided)
        # raise NotImplementedError()
        
        padding_location = (X != self.embedding_layer.padding_idx).unsqueeze(1) #.unsqueeze(1).repeat(1, X.size(1), 1)
        # print("padding_loc:", padding_location.size())
        padding_mask = torch.einsum('bki, bkj -> bij', padding_location, padding_location)        
        
        # print("without_repeat:", (X != 0).size())
        # print("with repeat:", (X == 0).unsqueeze(1).size())#.repeat(1, X.size(1), 1).size())
        # print("padd_mask:", padding_mask.size())

        embed_X = self.embedding_layer(X)
        # print("X after embedding:", embed_X.size())
        
        X = self.position_encoding(embed_X)
        # print("X after positional encoding:", X.size())
        
        # print("X:", X.size(), "padding_mask:", padding_mask.size())
        # padding_mask = padding_mask.repeat(1, 1, X.size(1), 1)
        
        for vi, layer in enumerate(self.blocks):
            # print("layer num:", vi)            
            # print("X:", X.size(), "padding_mask:", padding_mask.size())
            X = layer(X, padding_mask)            

        return X, padding_location.squeeze(1)
        

class DecoderBlock(Module):

    def __init__(self, embedding_dim, n_heads) -> None:
        super().__init__()

        self.attention1 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)
        self.attention2 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dropout3 = Dropout(0.2)

    def forward(self, encoded_source: torch.Tensor, target: torch.Tensor,
                mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of a decoder block. encoded_source has dimensions (batch_size, source_sequence_length, embedding_dim)
        and target has dimensions (batch_size, target_sequence_length, embedding_dim).

        The mask1 is passed to the first multi-headed self-attention layer, and mask2 is passed
        to the second multi-headed self-attention layer.

        Returns its output of shape (batch_size, target_sequence_length, embedding_dim) and
        the attention matrices for each of the heads of the second multi-headed self-attention layer
        (the one where the source and target are "mixed").
        """
        att_out, _ = self.attention1(target, target, target, mask1)
        residual = target + self.dropout1(att_out)

        X = self.norm1(residual)

        att_out, att_weights = self.attention2(X, encoded_source, encoded_source, mask2)
        
        residual = X + self.dropout2(att_out)
        X = self.norm2(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)
        residual = X + self.dropout3(temp)

        return self.norm3(residual), att_weights

class Decoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([DecoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])

        self.linear = Linear(embedding_dim, vocab_size + 1)
        self.softmax = Softmax(-1)

        self.vocab_size = vocab_size

    def _lookahead_mask(self, seq_length: int) -> torch.Tensor:
        """
        Compute the mask to prevent the decoder from looking at future target values.
        The mask you return should be a tensor of shape (sequence_length, sequence_length)
        with only 1's and 0's, where a 0 represent an entry that will be masked in the
        multi-headed attention layer.

        Hint: The function torch.tril (https://pytorch.org/docs/stable/generated/torch.tril.html)
        may be useful.
        """
        # TODO: Implement the lookahead mask
        # raise NotImplementedError()
        
        mask = torch.ones(seq_length, seq_length)
        mask = torch.tril(mask, diagonal=0)
        
        return mask


    def forward(self, encoded_source: torch.Tensor, source_padding: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Transformer decoder. encoded_source has dimensions (batch_size, source_sequence_length, embedding),
        source_padding has dimensions (batch_size, source_seuqence_length) and target has dimensions
        (batch_size, target_sequence_length).

        Returns its output of shape (batch_size, target_sequence_length, target_vocab_size) and
        the attention weights from the first decoder block, of shape
        (batch_size, n_heads, source_sequence_length, target_sequence_length)

        Note that the output is not normalized (i.e. we don't use the softmax function).
        """
        # TODO: Implement the decoder (you should re-use the DecoderBlock class that we provided)
        # raise NotImplementedError()
        
        target_padding_location = (target != self.embedding_layer.padding_idx).unsqueeze(1)

        # print("target_padding_location:", target_padding_location.size(), source_padding.size())
        
        padding_mask = torch.einsum('bki, bkj -> bij', target_padding_location, source_padding.unsqueeze(1))
        mask2 = padding_mask
        
        # print("mask2:", mask2[0, :10, :10])
        
        target_lookahead_mask = self._lookahead_mask(target.size(1))
        target_padding_mask = torch.einsum('bki, bkj -> bij', target_padding_location, target_padding_location)        
        # target_mask = target_padding_location.repeat(1, target.size(1), 1) * target_lookahead_mask.unsqueeze(0).repeat(target.size(0), 1, 1)
        target_mask = torch.min(target_padding_mask, target_lookahead_mask.unsqueeze(0))
        mask1 = target_mask
        
        embed_target = self.embedding_layer(target)
        position_target = self.position_encoding(embed_target)
        
        # print("target_padding_mask dimen:",  target_padding_location.size())
         #.unsqueeze(1).repeat(1, X.size(1), 1)
        # print("target_lookahead_mask:", target_lookahead_mask.size())
        # print("target_mask:", target_mask.size())        
        
        # X = position_target
        for vi, layer in enumerate(self.blocks):
            # print("layer num:", vi, "mask1:", mask1.size(), "mask2:", mask2.size())            
            if vi == 0:
                position_target, attention_weights1 = layer(encoded_source, position_target, mask1, mask2) 
            else:
                position_target, attention_weights = layer(encoded_source, position_target, mask1, mask2) 
        
        output = self.linear(position_target)
        # print(output.shape, attention_weights.shape)
        
        return output, attention_weights1

class Transformer(Module):

    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int, n_encoder_blocks: int,
                 n_decoder_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.encoder = Encoder(source_vocab_size, embedding_dim, n_encoder_blocks, n_heads)
        self.decoder = Decoder(target_vocab_size, embedding_dim, n_decoder_blocks, n_heads)
        self.tgt_vocab = target_vocab_size

    def forward(self, source, target):
        encoded_source, source_padding = self.encoder(source)
        return self.decoder(encoded_source, source_padding, target)
    
    def predict(self, source: List[int], beam_size=1, max_length=64) -> List[int]:
            """
            Given a sentence in the source language, you should output a sentence in the target
            language of length at most `max_length` that you generate using a beam search with
            the given `beam_size`.
    
            Note that the start of sentence token is 2 and the end of sentence token is 3.
    
            Return the final top beam (decided using average log-likelihood) and its average
            log-likelihood.
    
            Hint: The follow functions may be useful:
                - torch.topk (https://pytorch.org/docs/stable/generated/torch.topk.html)
                - torch.softmax (https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
            """
            self.eval() # Set the PyTorch Module to inference mode (this affects things like dropout)
            SOS = 2
            EOS = 3
            if not isinstance(source, torch.Tensor):
                source_input = torch.tensor(source).view(1, -1)
            else:
                source_input = source.view(1, -1)
            
            encoded_source_ini, source_padding = self.encoder(source_input)
            decoder_input = [torch.LongTensor([[SOS]]) for ii in range(beam_size)]
            scores = [torch.tensor([0]) for ii in range(beam_size)]
            curr_beam = beam_size
        
    
            completed_decode = []
            completed_score = []
    
            for i in range(max_length-1):
                if curr_beam>1:
                    encoded_source = encoded_source_ini.expand(curr_beam, *encoded_source_ini.shape[1:])
                else:
                    encoded_source, source_padding = self.encoder(source_input)
                # Decoder prediction
                print(decoder_input.size())
                logits, _ = self.decoder(encoded_source,source_padding,torch.cat(decoder_input))
                
    
                # Softmax
                log_probs = torch.log_softmax(logits[:, -1], 1)
                
                if i == 0:
                    a, b = torch.topk(log_probs+torch.cat(scores).view(-1,1), curr_beam)
                    k_token = b[0,:]
                    k_score = a[0,:]
                    beam_indices  = torch.arange(0,curr_beam)
                else:
                    k_score, k_token = torch.topk((log_probs+torch.cat(scores).view(-1,1)).view(-1), curr_beam)
                    beam_indices  = torch.divide   (k_token, self.tgt_vocab+1, rounding_mode='floor') # indices // vocab_size
                token_indices = torch.remainder(k_token, self.tgt_vocab+1)    
                temp_decodes = []
                for idx, best_beam_idx in enumerate(beam_indices):
                    prev_inp = decoder_input[best_beam_idx]
                    token_index = torch.LongTensor([[token_indices[idx]]])
    
                    prev_inp = torch.cat([prev_inp, token_index],dim=1)
    
                    temp_decodes.append(prev_inp)
                    scores[idx] = k_score[idx].view(-1,1)
                    
                decoder_input = temp_decodes
    
    
                done_seq=[]
                for eos_check in range(len(decoder_input)):
                    if decoder_input[eos_check][:,-1] == EOS:
                        done_seq.append(eos_check)
                for del_idx in range(len(done_seq)):
                    completed_decode.append(decoder_input.pop(sorted(done_seq,reverse=True)[del_idx]))
                    completed_score.append(scores.pop(sorted(done_seq,reverse=True)[del_idx]).detach())
                    curr_beam -= 1
    
                if len(decoder_input) == 0:
                    break
                # print(torch.cat(scores).T/(i+2))
                # print(torch.vstack(decoder_input))
                if i == max_length-2:
                    done_idx = np.argmax(torch.cat(scores).detach())
                    completed_decode.append(decoder_input.pop(done_idx))
                    completed_score.append(scores.pop(done_idx).detach())
                    curr_beam -= 1
    
            # calc_score = np.zeros(len(completed_score))
            # for i in range(len(completed_score)):
            #     calc_score[i] = completed_score[i]/len(completed_decode[i][0])
    
            calc_score = np.zeros(len(completed_score))
            for i in range(len(completed_decode)):
                calc_score[i] = completed_score[i]/(len(completed_decode[i][0]))
            best_idx = np.argmax(calc_score)
            return completed_decode[best_idx].detach().flatten(), completed_score[best_idx]/len(completed_decode[best_idx][0])                

def load_data() -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]]:
    """ Load the dataset.

    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('data/translation_data.bin', 'rb') as f:
        corpus, source_vocab, target_vocab = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:]
        print("# source vocab: {}\n"
              "# target vocab: {}\n"
              "# train sentences: {}\n"
              "# test sentences: {}\n".format(len(source_vocab), len(target_vocab), len(train_sentences),
                                              len(test_sentences)))
        return train_sentences, test_sentences, source_vocab, target_vocab

def preprocess_data(sentences: Tuple[List[int], List[int]], source_vocab_size,
                    target_vocab_size, max_length):

    source_sentences = []
    target_sentences = []

    for source, target in sentences:
        source = [0] + source + ([source_vocab_size] * (max_length - len(source) - 1))
        target = [0] + target + ([target_vocab_size] * (max_length - len(target) - 1))
        source_sentences.append(source)
        target_sentences.append(target)

    return torch.tensor(source_sentences), torch.tensor(target_sentences)

def decode_sentence(encoded_sentence: List[int], vocab: Dict) -> str:
    if isinstance(encoded_sentence, torch.Tensor):
        encoded_sentence = [w.item() for w in encoded_sentence]
    words = [vocab[w] for w in encoded_sentence if w != 0 and w != 1 and w in vocab]
    return " ".join(words)

def visualize_attention(source_sentence: List[int],
                        output_sentence: List[int],
                        source_vocab: Dict[int, str],
                        target_vocab: Dict[int, str],
                        attention_matrix: np.ndarray):
    """
    :param source_sentence_str: the source sentence, as a list of ints
    :param output_sentence_str: the target sentence, as a list of ints
    :param attention_matrix: the attention matrix, of dimension [target_sentence_len x source_sentence_len]
    :param outfile: the file to output to
    """
    source_length = 0
    while source_length < len(source_sentence) and source_sentence[source_length] != 3:
        source_length += 1

    target_length = 0
    while target_length < len(output_sentence) and output_sentence[target_length] != 3:
        target_length += 1

    source_length += 1
    target_length += 1

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix[:target_length, :source_length], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(source_length)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in source_vocab else source_vocab[x] for x in source_sentence[:source_length]]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(target_length)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in target_vocab else target_vocab[x] for x in output_sentence[:target_length]]))

    plt.show()
    plt.close()

def train(model: Transformer, train_source: torch.Tensor, train_target: torch.Tensor,
          test_source: torch.Tensor, test_target: torch.Tensor, target_vocab_size: int,
          epochs: int = 30, batch_size: int = 64, lr: float = 0.0001):

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss(ignore_index=target_vocab_size)

    epoch_train_loss = np.zeros(epochs)
    epoch_test_loss = np.zeros(epochs)

    for ep in range(epochs):

        train_loss = 0
        test_loss = 0

        permutation = torch.randperm(train_source.shape[0])
        train_source = train_source[permutation]
        train_target = train_target[permutation]

        batches = train_source.shape[0] // batch_size
        model.train()
        for ba in tqdm(range(batches), desc=f"Epoch {ep + 1}"):

            optimizer.zero_grad()

            batch_source = train_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = train_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        test_batches = test_source.shape[0] // batch_size
        model.eval()
        for ba in tqdm(range(test_batches), desc="Test", leave=False):

            batch_source = test_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = test_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            test_loss += batch_loss.item()

        epoch_train_loss[ep] = train_loss / batches
        epoch_test_loss[ep] = test_loss / test_batches
        print(f"Epoch {ep + 1}: Train loss = {epoch_train_loss[ep]:.4f}, Test loss = {epoch_test_loss[ep]:.4f}")

    return epoch_train_loss, epoch_test_loss

def bleu_score(predicted: List[int], target: List[int], N: int = 4) -> float:
    """
    Implement a function to compute the BLEU-N score of the predicted
    sentence with a single reference (target) sentence.

    Please refer to the handout for details.

    Make sure you strip the SOS (2), EOS (3), and padding (anything after EOS)
    from the predicted and target sentences.

    If the length of the predicted sentence or the target is less than N,
    the BLEU score is 0.
    """
    # TODO: Implement bleu score
    # raise NotImplementedError()
    predicted_sentence = predicted[1:predicted.index(3)]
    target_sentence = target[1:target.index(3)]
    precision_all = []
    
    if len(predicted_sentence) < N or len(target_sentence) < N:
        return 0
    
    for sub_n_gram in range(1, N+1):
        
        predicted_ngrams = {}
       
        for i in range(len(predicted_sentence)-sub_n_gram+1):
            ngram = tuple(predicted_sentence[i:i+sub_n_gram])
            predicted_ngrams[ngram] = predicted_ngrams.get(ngram, 0) + 1
            
        target_ngrams = {}
        for i in range(len(target_sentence)-sub_n_gram+1):
            ngram = tuple(target_sentence[i:i+sub_n_gram])
            target_ngrams[ngram] = target_ngrams.get(ngram, 0) + 1
            
        intersection_counts = {}
        for k in predicted_ngrams.keys():
            if k in target_ngrams:
                intersection_counts[k] = min(predicted_ngrams[k], target_ngrams[k])    
        
        tot_value = sum(intersection_counts.values())
        
        precision = tot_value / (len(predicted_sentence) - sub_n_gram + 1)
        
        precision_all.append(precision ** (1/N))
        
    prod = np.prod(precision_all)
    
    length_penalty = min(1, np.exp(1 - len(target_sentence) / len(predicted_sentence)))
    
    return prod * length_penalty



def seed_everything(seed=10707):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Loads data from English -> Spanish machine translation task
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    max_length = 64
    q_no = "2e"
    # Generates train/test data based on english and french vocabulary sizes and caps max length of sentence at 64
    train_source, train_target = preprocess_data(train_sentences, len(source_vocab), len(target_vocab), max_length)
    test_source, test_target = preprocess_data(test_sentences, len(source_vocab), len(target_vocab), max_length)
    source_vocab_size, target_vocab_size = len(source_vocab), len(target_vocab)
    embedding_dim = 256
    n_encoder_blocks, n_decoder_blocks, n_heads = 2, 4, 3
    model = Transformer(source_vocab_size, target_vocab_size, embedding_dim, n_encoder_blocks, n_decoder_blocks, n_heads)
    # torch.save(model.state_dict(), "model_"+q_no+".pkl")
    # epoch_train_loss, epoch_test_loss = train(model, train_source, train_target, test_source, test_target, target_vocab_size)
    # np.save("epoch_train_loss_"+q_no+".npy", epoch_train_loss)
    # np.save("epoch_test_loss_"+q_no+".npy", epoch_test_loss)
    # plt.figure()
    # plt.plot(epoch_train_loss, label='epoch_train_loss')
    # plt.plot(epoch_test_loss, label='epoch_test_loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.savefig(q_no+".png")
    # plt.show()
    #----------------------
    model.load_state_dict(torch.load("model_"+q_no+".pkl"))
    
    print(test_source.shape)
    
    for i in range(8):
        words = decode_sentence(test_source[i, :], source_vocab)
        print("source")
        print(words)
    
        words = decode_sentence(test_target[i, :], target_vocab)
        print("target")
        print(words)    
        
        translated_sentence, score = model.predict(test_source[i, :], beam_size=3, max_length=64)
        
        words = decode_sentence(translated_sentence, target_vocab)
        print("predicted")
        print(words)
        print("score:", score)
    #----------------------
    # model.load_state_dict(torch.load("model_"+q_no+".pkl"))
    # for i in range(3):
    #     _, att_weights = model.forward(train_source[i, :].reshape(1, -1), train_target[i, :].reshape(1, -1))
    #     s1, s2, s3, s4 = att_weights.shape
    #     print("att_weights:", att_weights.reshape(s3, s4).shape)
    #     visualize_attention(train_source[i, :], train_target[i, :], source_vocab, target_vocab, att_weights.reshape(s3, s4).detach().numpy())
