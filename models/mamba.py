import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from module.Attention import *
from module.CPUEmbedding import *
from module.Common import *
from models.moe import LatentMoEClassifier

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None


def _resolve_mamba_cls(variant):
    variant = variant.lower()
    if variant == 'auto':
        if Mamba is not None:
            return Mamba
        if Mamba2 is not None:
            return Mamba2
    elif variant == 'mamba2':
        if Mamba2 is not None:
            return Mamba2
    elif variant == 'mamba':
        if Mamba is not None:
            return Mamba
    raise ImportError(
        'Bidirectional Mamba requires mamba-ssm. '
        'Install a compatible PyTorch/CUDA stack first, then install mamba-ssm.'
    )


class BidirectionalMambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.0, variant='auto'):
        super(BidirectionalMambaLayer, self).__init__()
        mamba_cls = _resolve_mamba_cls(variant)
        self.norm = nn.LayerNorm(d_model)
        self.forward_block = mamba_cls(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_block = mamba_cls(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = dropout

    def forward(self, x, masks=None):
        residual = x
        x = self.norm(x)
        y_forward = self.forward_block(x)
        y_backward = torch.flip(self.backward_block(torch.flip(x, dims=(1,))), dims=(1,))
        y = 0.5 * (y_forward + y_backward)
        if masks is not None:
            y = y * masks.unsqueeze(-1).type_as(y)
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)
        return residual + y


class AttBiMambaModel(nn.Module):
    _logger = logging.getLogger('AttBiMamba')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'AttBiMamba.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for Attention-Based BiMamba succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return AttBiMambaModel._logger

    def __init__(self, vocab, lstm_layers, lstm_hiddens, dropout=0, mamba_state=64, mamba_conv=4,
                 mamba_expand=2, mamba_variant='auto', use_moe=False, moe_num_experts=4, moe_top_k=2,
                 moe_bottleneck_dim=None, moe_temperature=1.5, moe_gate_dropout=0.1,
                 moe_balance_loss_weight=1e-2, moe_diversity_loss_weight=1e-3, moe_z_loss_weight=0.0):
        super(AttBiMambaModel, self).__init__()
        self.dropout = dropout
        self.use_moe = use_moe
        vocab_size, word_dims = vocab.vocab_size, vocab.word_dim
        self.word_embed = CPUEmbedding(vocab_size, word_dims, padding_idx=vocab_size - 1)
        self.word_embed.weight.data.copy_(torch.from_numpy(vocab.embeddings))
        self.word_embed.weight.requires_grad = False

        self.sent_dim = 2 * lstm_hiddens
        self.input_proj = nn.Linear(word_dims, self.sent_dim) if word_dims != self.sent_dim else nn.Identity()
        self.layers = nn.ModuleList([
            BidirectionalMambaLayer(
                d_model=self.sent_dim,
                d_state=mamba_state,
                d_conv=mamba_conv,
                expand=mamba_expand,
                dropout=dropout,
                variant=mamba_variant,
            )
            for _ in range(lstm_layers)
        ])

        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)
        if self.use_moe:
            self.proj = LatentMoEClassifier(
                input_dim=self.sent_dim,
                output_dim=2,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                bottleneck_dim=moe_bottleneck_dim,
                temperature=moe_temperature,
                gate_dropout=moe_gate_dropout,
                balance_loss_weight=moe_balance_loss_weight,
                diversity_loss_weight=moe_diversity_loss_weight,
                z_loss_weight=moe_z_loss_weight,
            )
        else:
            self.proj = NonLinear(self.sent_dim, 2)

        self.logger.info('==== Model Parameters ====')
        self.logger.info('Input Dimension: %d' % word_dims)
        self.logger.info('Hidden Size: %d' % lstm_hiddens)
        self.logger.info('Num Layers: %d' % lstm_layers)
        self.logger.info('Dropout %.3f' % dropout)
        self.logger.info('Mamba Variant: %s' % mamba_variant)
        self.logger.info('Mamba State: %d' % mamba_state)
        self.logger.info('Mamba Conv: %d' % mamba_conv)
        self.logger.info('Mamba Expand: %d' % mamba_expand)
        if self.use_moe:
            self.logger.info('MoE Enabled: experts=%d, top_k=%d, bottleneck_dim=%s, temperature=%.3f, '
                             'balance_weight=%.4g, diversity_weight=%.4g, z_weight=%.4g'
                             % (
                                 moe_num_experts,
                                 min(moe_top_k, moe_num_experts),
                                 str(moe_bottleneck_dim if moe_bottleneck_dim is not None else max(self.sent_dim // 4, 1)),
                                 moe_temperature,
                                 moe_balance_loss_weight,
                                 moe_diversity_loss_weight,
                                 moe_z_loss_weight,
                             ))

    def reset_word_embed_weight(self, vocab, pretrained_embedding):
        vocab_size, word_dims = pretrained_embedding.shape
        self.word_embed = CPUEmbedding(vocab.vocab_size, word_dims, padding_idx=vocab.PAD)
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

    def forward(self, inputs):
        words, masks, word_len = inputs
        embed = self.word_embed(words)
        if self.training:
            embed = drop_input_independent(embed, self.dropout)
        embed = embed.cuda(device)

        hiddens = self.input_proj(embed)
        if masks is not None:
            hiddens = hiddens * masks.unsqueeze(-1).type_as(hiddens)
        for layer in self.layers:
            hiddens = layer(hiddens, masks)
        if masks is not None:
            hiddens = hiddens * masks.unsqueeze(-1).type_as(hiddens)

        batch_size = hiddens.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        represents = hiddens * sent_probs
        represents = represents.sum(dim=1)
        outputs = self.proj(represents)
        return outputs

    def get_auxiliary_loss(self):
        if self.use_moe:
            return self.proj.get_auxiliary_loss()
        return self.atten_guide.new_zeros(())

    def get_moe_metrics(self):
        if self.use_moe:
            return self.proj.get_metrics()
        return {}
