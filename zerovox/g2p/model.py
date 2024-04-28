#
# this code is based on DeepPhonemizer, license: MIT
#

import os
import math 
from enum import Enum
from typing import Tuple, Dict, Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from lightning import LightningModule

from .data import G2PSymbols

class ModelType(Enum):
    TRANSFORMER = 'transformer'
    AUTOREG_TRANSFORMER = 'autoreg_transformer'

    def is_autoregressive(self) -> bool:
        """
        Returns: bool: Whether the model is autoregressive.
        """
        return self in {ModelType.AUTOREG_TRANSFORMER}

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout=0.1, max_len=5000) -> None:
        """
        Initializes positional encoding.

        Args:
            d_model (int): Dimension of model.
            dropout (float): Dropout after positional encoding.
            max_len: Max length of precalculated position sequence.
        """

        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', torch.nn.Parameter(pe, requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # shape: [T, N]
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

# class ForwardTransformer(Model):

#     def __init__(self,
#                  encoder_vocab_size: int,
#                  decoder_vocab_size: int,
#                  d_model=512,
#                  d_fft=1024,
#                  layers=4,
#                  dropout=0.1,
#                  heads=1) -> None:
#         super().__init__()

#         self.d_model = d_model

#         self.embedding = nn.Embedding(encoder_vocab_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model, dropout)

#         encoder_layer = TransformerEncoderLayer(d_model=d_model,
#                                                 nhead=heads,
#                                                 dim_feedforward=d_fft,
#                                                 dropout=dropout,
#                                                 activation='relu')
#         encoder_norm = LayerNorm(d_model)
#         self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
#                                           num_layers=layers,
#                                           norm=encoder_norm)

#         self.fc_out = nn.Linear(d_model, decoder_vocab_size)

#     def forward(self,
#                 batch: Dict[str, torch.Tensor]) -> torch.Tensor:         # shape: [N, T]
#         """
#         Forward pass of the model on a data batch.

#         Args:
#          batch (Dict[str, torch.Tensor]): Input batch entry 'text' (text tensor).

#         Returns:
#           Tensor: Predictions.
#         """

#         x = batch['text']
#         x = x.transpose(0, 1)        # shape: [T, N]
#         src_pad_mask = _make_len_mask(x).to(x.device)
#         x = self.embedding(x)
#         x = self.pos_encoder(x)
#         x = self.encoder(x, src_key_padding_mask=src_pad_mask)
#         x = self.fc_out(x)
#         x = x.transpose(0, 1)
#         return x

#     @torch.jit.export
#     def generate(self,
#                  batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Inference pass on a batch of tokenized texts.

#         Args:
#           batch (Dict[str, torch.Tensor]): Input batch with entry 'text' (text tensor).

#         Returns:
#           Tuple: The first element is a Tensor (phoneme tokens) and the second element
#                  is a tensor (phoneme token probabilities).
#         """

#         with torch.no_grad():
#             x = self.forward(batch)
#         tokens, logits = get_dedup_tokens(x)
#         return tokens, logits



def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def _make_len_mask(inp: torch.Tensor, pad_index: int) -> torch.Tensor:
    return (inp == pad_index).transpose(0, 1)

class AutoregressiveTransformer(torch.nn.Module):

    def __init__(self,
                 symbols: G2PSymbols,
                 d_model=512,
                 d_fft=1024,
                 encoder_layers=4,
                 decoder_layers=4,
                 dropout=0.1,
                 heads=1):
        
        super().__init__()

        self._symbols = symbols
        self.d_model = d_model
        self.encoder = torch.nn.Embedding(symbols.num_graphemes, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.decoder = torch.nn.Embedding(symbols.num_phonemes, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = torch.nn.Transformer(d_model=d_model, nhead=heads, num_encoder_layers=encoder_layers,
                                                num_decoder_layers=decoder_layers, dim_feedforward=d_fft,
                                                dropout=dropout, activation='relu')
        self.fc_out = torch.nn.Linear(d_model, symbols.num_phonemes)

    def __call__(self, src: torch.Tensor, trg: torch.Tensor):
        """
        Foward pass of the model on a data batch.
        """

        trg = trg[:, :-1] # FIXME: why?

        src = src.transpose(0, 1)        # shape: [T, N]
        trg = trg.transpose(0, 1)

        trg_mask = _generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = _make_len_mask(src, self._symbols.pad_token_gidx).to(src.device)
        trg_pad_mask = _make_len_mask(trg, self._symbols.pad_token_pidx).to(trg.device)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=None, tgt_mask=trg_mask,
                                  memory_mask=None, src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)
        output = output.transpose(0, 1)
        return output

    def generate(self,
                 input: torch.Tensor,
                 max_len: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference pass on a batch of tokenized texts.

        Args:
          batch (torch.Tensor): tensor of batch of input token indices
          max_len (int): Max steps of the autoregressive inference loop.

        Returns:
          Tuple: Predictions. The first element is a Tensor of phoneme tokens and the second element
                 is a Tensor of phoneme token probabilities.
        """

        batch_size = input.size(0)
        input = input.transpose(0, 1)          # shape: [T, N]
        src_pad_mask = _make_len_mask(input, self._symbols.pad_token_gidx).to(input.device)
        with torch.no_grad():
            input = self.encoder(input)
            input = self.pos_encoder(input)
            input = self.transformer.encoder(input,
                                             src_key_padding_mask=src_pad_mask)
            out_indices = torch.tensor([[self._symbols.start_token_pidx] * batch_size], dtype=torch.long)
            out_logits = []
            for i in range(max_len):
                tgt_mask = _generate_square_subsequent_mask(i + 1).to(input.device)
                output = self.decoder(out_indices)
                output = self.pos_decoder(output)
                output = self.transformer.decoder(output,
                                                  input,
                                                  memory_key_padding_mask=src_pad_mask,
                                                  tgt_mask=tgt_mask)
                output = self.fc_out(output)  # shape: [T, N, V]
                out_tokens = output.argmax(2)[-1:, :]
                out_logits.append(output[-1:, :, :])

                out_indices = torch.cat([out_indices, out_tokens], dim=0)
                stop_rows, _ = torch.max(out_indices == self._symbols.end_token_pidx, dim=0)
                if torch.sum(stop_rows) == batch_size:
                    break

        out_indices = out_indices.transpose(0, 1)  # out shape [N, T]
        out_logits = torch.cat(out_logits, dim=0).transpose(0, 1) # out shape [N, T, V]
        out_logits = out_logits.softmax(-1)
        out_probs = torch.ones((out_indices.size(0), out_indices.size(1)))
        for i in range(out_indices.size(0)):
            for j in range(0, out_indices.size(1)-1):
                out_probs[i, j+1] = out_logits[i, j].max()
        return out_indices, out_probs

    @classmethod
    def from_config(cls, config: Dict[str, Any], symbols: G2PSymbols) -> 'AutoregressiveTransformer':
        """
        Initializes an autoregressive Transformer model from a config.
        Args:
          config (dict): Configuration containing the hyperparams.

        Returns:
          AutoregressiveTransformer: Model object.
        """

        #preprocessor = Preprocessor.from_config(config)
        return AutoregressiveTransformer(symbols,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            encoder_layers=config['model']['layers'],
            decoder_layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )

class CrossEntropyLoss(torch.nn.Module):
    """ """

    def __init__(self) -> None:
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self,
                pred: torch.Tensor,
                target) -> torch.Tensor:
        
        p = pred.transpose(1, 2)
        t = target[:, 1:] # FIXME: why?

        loss = self.criterion(p, t)
        return loss


class CTCLoss(torch.nn.Module):
    """ """

    def __init__(self):
        super().__init__()
        self.criterion  = torch.nn.CTCLoss()

    def forward(self,
                pred: torch.Tensor,
                batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the CTCLoss module on a batch.

        Args:
          pred: Batch of model predictions.
          batch: Dictionary of a training data batch, containing 'phonemes': target phonemes,
        'text_len': input text lengths, 'phonemes_len': target phoneme lengths
          pred: torch.Tensor:
          batch: Dict[str: 
          torch.Tensor]:

        Returns:
          Loss as tensor.

        """

        pred = pred.transpose(0, 1).log_softmax(2)
        phonemes = batch['phonemes']
        text_len = batch['text_len']
        phon_len = batch['phonemes_len']
        loss = self.criterion(pred, phonemes, text_len, phon_len)
        return loss

def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=0):
    """
    Create a learning rate scheduler with linear warm-up and cosine learning rate decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to create the scheduler.
        warmup_steps (int): The number of warm-up steps.
        total_steps (int): The total number of steps.
        min_lr (float, optional): The minimum learning rate at the end of the decay. Default: 0.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.
    """

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warm-up
            f = float(current_epoch+1) / float(warmup_epochs)
        else:
            # Cosine learning rate decay
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            f = max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return f

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

class LightningTransformer(LightningModule):

    def __init__(self, model_type: ModelType, config: Dict[str, Any], symbols: G2PSymbols, val_dir: os.PathLike, lr: float,
                 weight_decay:float, max_epochs:int, warmup_epochs:int):
        super().__init__()

        self.save_hyperparameters()

        self._config        = config
        self._lr            = lr
        self._val_dir       = val_dir
        self._symbols       = symbols
        self._weight_decay  = weight_decay
        self._max_epochs    = max_epochs
        self._warmup_epochs = warmup_epochs

        if model_type is ModelType.TRANSFORMER:
            # FIXME model = ForwardTransformer.from_config(config, symbols)
            assert False
        elif model_type is ModelType.AUTOREG_TRANSFORMER:
            self._model = AutoregressiveTransformer.from_config(config, symbols)
        else:
            raise ValueError(f'Unsupported model type: {model_type}. Supported types: {[t.value for t in ModelType]}')
        
        if model_type.is_autoregressive():
            self._loss = CrossEntropyLoss()
        else:
            self._loss = CTCLoss()

        self._training_step_outputs = []
        self._validation_step_outputs = []

    def forward(self, inputs, target):
        return self._model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs = batch['graph_idxs']
        target = batch['phone_idxs']
        prediction = self(inputs, target)
        loss = self._loss(prediction, target)

        self._training_step_outputs.append(loss)

        return loss
    
    def validation_step(self, batch, batch_idx):

        if self.current_epoch>=1 :

            inputs = batch['graph_idxs']
            target = batch['phone_idxs']
            prediction = self(inputs, target)

            os.makedirs(self._val_dir, exist_ok=True)

            # write validation targets and preditions to file
            path = self._val_dir / f'val_{self.current_epoch}.txt'
            with open(path, "a") as f:
                for inp, tar, pred in zip(inputs, target, prediction):

                    word = self._symbols.convert_ids_to_graphemes(inp.tolist())
                    phone_target = self._symbols.convert_ids_to_phonemes(tar.tolist())
                    phone_pred = self._symbols.convert_ids_to_phonemes(pred.argmax(1).tolist())

                    f.write(f'{"".join(word[1:])}: {"".join(phone_pred)} vs {"".join(phone_target)}\n')

            loss = self._loss(prediction, target)

            self._validation_step_outputs.append(loss)

            return loss

        return torch.tensor(0.0)

    def configure_optimizers(self):
        optimizer = AdamW(self._model.parameters(), self._lr, weight_decay=self._weight_decay)
        self.scheduler = get_lr_scheduler(optimizer, warmup_epochs=self._warmup_epochs, total_epochs=self._max_epochs, min_lr=0.1)
        return [optimizer], [self.scheduler]

    def on_train_epoch_end(self):
        if not self._training_step_outputs:
            return
        avg_loss = torch.stack(self._training_step_outputs).mean()
        self.log("loss", avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", self.scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True)
        self._training_step_outputs.clear()

        if not self._validation_step_outputs:
            return
        avg_loss = torch.stack(self._validation_step_outputs).mean()
        self.log("val", avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        self._validation_step_outputs.clear()

    def generate(self,
                 input: torch.Tensor,
                 max_len: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference pass on a batch of tokenized texts.

        Args:
          batch (torch.Tensor): tensor of batch of input token indices
          start_token_pidx: start token phoneme index (from symbols)
          max_len (int): Max steps of the autoregressive inference loop.

        Returns:
          Tuple: Predictions. The first element is a Tensor of phoneme tokens and the second element
                 is a Tensor of phoneme token probabilities.
        """
        return self._model.generate(input, max_len=max_len)