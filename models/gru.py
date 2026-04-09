from module.Attention import *
from module.CPUEmbedding import *
from module.Common import *
from models.moe import LatentMoEClassifier


class AttGRUModel(nn.Module):
    # Dispose Loggers.
    _logger = logging.getLogger('AttGRU')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'AttGRU.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for Attention-Based GRU succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return AttGRUModel._logger

    def __init__(self, vocab, lstm_layers, lstm_hiddens, dropout=0, use_moe=False, moe_num_experts=4,
                 moe_top_k=2, moe_bottleneck_dim=None, moe_temperature=1.5, moe_gate_dropout=0.1,
                 moe_balance_loss_weight=1e-2, moe_diversity_loss_weight=1e-3, moe_z_loss_weight=0.0):
        super(AttGRUModel, self).__init__()
        self.dropout = dropout
        self.use_moe = use_moe
        self.logger.info('==== Model Parameters ====')
        vocab_size, word_dims = vocab.vocab_size, vocab.word_dim
        self.word_embed = CPUEmbedding(vocab_size, word_dims, padding_idx=vocab_size - 1)
        self.word_embed.weight.data.copy_(torch.from_numpy(vocab.embeddings))
        self.word_embed.weight.requires_grad = False
        self.logger.info('Input Dimension: %d' % word_dims)
        self.logger.info('Hidden Size: %d' % lstm_hiddens)
        self.logger.info('Num Layers: %d' % lstm_layers)
        self.logger.info('Dropout %.3f' % dropout)
        self.rnn = nn.GRU(input_size=word_dims, hidden_size=lstm_hiddens, num_layers=lstm_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.sent_dim = 2 * lstm_hiddens
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
        else:
            self.proj = NonLinear(self.sent_dim, 2)

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
        embed = embed.to(device)
        batch_size = embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        hiddens, state = self.rnn(embed)
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        represents = hiddens * sent_probs
        # represents = hiddens
        represents = represents.sum(dim=1)
        # represents = represents[:, -1, :]
        outputs = self.proj(represents)
        return outputs  # , represents

    def get_auxiliary_loss(self):
        if self.use_moe:
            return self.proj.get_auxiliary_loss()
        return self.atten_guide.new_zeros(())

    def get_moe_metrics(self):
        if self.use_moe:
            return self.proj.get_metrics()
        return {}

    def backbone_parameters(self):
        params = list(self.word_embed.parameters())
        params.extend(self.rnn.parameters())
        params.append(self.atten_guide)
        params.extend(self.atten.parameters())
        return params

    def gate_parameters(self):
        if not self.use_moe:
            return list(self.proj.parameters())
        params = list(self.proj.input_norm.parameters())
        params.extend(self.proj.router.parameters())
        return params

    def expert_parameters(self):
        if not self.use_moe:
            return []
        params = list(self.proj.down_projs.parameters())
        params.extend(self.proj.up_projs.parameters())
        params.extend(self.proj.heads.parameters())
        return params
