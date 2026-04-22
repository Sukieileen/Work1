import sys

sys.path.extend([".", ".."])

import argparse
import csv
import math
from dataclasses import dataclass

from CONSTANTS import *
from entities.TensorInstances import TInstWithLogits
from entities.instances import Instance
from models.mamba import AttBiMambaModel
from preprocessing.Preprocess import Preprocessor
from representations.parser_free import ParserFreeEncoder
from utils.Vocab import Vocab

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    average_precision_score = None
    roc_auc_score = None


lstm_hiddens = 100
num_layer = 2
mamba_state = 64
mamba_conv = 4
mamba_expand = 2
mamba_variant = 'auto'


@dataclass
class DirectionConfig:
    name: str
    source_dataset: str
    target_dataset: str


DIRECTION_CONFIGS = {
    'hdfs_to_bgl': DirectionConfig(
        name='hdfs_to_bgl',
        source_dataset='HDFS',
        target_dataset='BGL',
    ),
    'bgl_to_hdfs': DirectionConfig(
        name='bgl_to_hdfs',
        source_dataset='BGL',
        target_dataset='HDFS',
    ),
}


CLEAN_DIRECTION_CONFIGS = {
    'hdfs_to_bgl': {
        'source_ratio': 0.3,
        'target_normal_ratio': 0.3,
        'target_anomaly_ratio': 0.01,
    },
    'bgl_to_hdfs': {
        'source_ratio': 1.0,
        'target_normal_ratio': 0.1,
        'target_anomaly_ratio': 0.01,
    },
}


PARSER_FREE_DEFAULTS = {
    'model_name': 'bert-base-uncased',
    'max_length': 64,
    'batch_size': 64,
    'pooling': 'mean',
    'cache_dir': '',
}


def sanitize_probs(tag_logits):
    tag_probs = F.softmax(tag_logits, dim=1)
    tag_probs = torch.nan_to_num(tag_probs, nan=0.5, posinf=1.0, neginf=0.0)
    tag_probs = torch.clamp(tag_probs, min=1e-6, max=1 - 1e-6)
    return tag_probs


def get_protocol_option(args, option_name, default_value):
    if args is None:
        return default_value
    return getattr(args, option_name, default_value)


def identity_cut(instances):
    return instances, [], []


def append_epoch_metrics(csv_file, row):
    fieldnames = [
        'direction',
        'phase',
        'epoch',
        'phase_mean_loss',
        'selected_threshold',
        'selection_f1',
        'test_precision',
        'test_recall',
        'test_f1',
        'test_auroc',
        'test_aucpr',
        'selected_for_best',
    ]
    output_dir = os.path.dirname(csv_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', encoding='utf-8', newline='') as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        if write_header:
            csv_writer.writeheader()
        csv_writer.writerow(row)


def load_checkpoint_state_dict(checkpoint_path):
    load_kwargs = {'map_location': device}
    try:
        return torch.load(checkpoint_path, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(checkpoint_path, **load_kwargs)


def clone_instance_with_sequence(instance, sequence):
    new_instance = Instance(instance.id, sequence, instance.label)
    new_instance.predicted = instance.predicted
    new_instance.confidence = instance.confidence
    new_instance.repr = instance.repr
    return new_instance


def build_merged_embeddings(domain_embeddings):
    merged_embeddings = {}
    domain_mappings = {}
    next_template_id = 1
    for domain_name, id2embed in domain_embeddings.items():
        mapping = {}
        for template_id in sorted(id2embed.keys()):
            mapping[template_id] = next_template_id
            merged_embeddings[next_template_id] = id2embed[template_id]
            next_template_id += 1
        domain_mappings[domain_name] = mapping
    return merged_embeddings, domain_mappings


def remap_instances(instances, mapping):
    remapped_instances = []
    for inst in instances:
        remapped_sequence = []
        for event_id in inst.sequence:
            if event_id not in mapping:
                raise KeyError('Missing event_id %s in merged embedding mapping.' % str(event_id))
            remapped_sequence.append(mapping[event_id])
        remapped_instances.append(clone_instance_with_sequence(inst, remapped_sequence))
    return remapped_instances


def split_instances_by_ratio(instances, ratio, rng):
    shuffled = list(instances)
    rng.shuffle(shuffled)
    train_size = int(len(shuffled) * ratio)
    return shuffled[:train_size], shuffled[train_size:]


def split_instances_by_sequence_groups(instances, ratio, rng):
    grouped_instances = {}
    for inst in instances:
        sequence_key = tuple(inst.sequence)
        if sequence_key not in grouped_instances:
            grouped_instances[sequence_key] = []
        grouped_instances[sequence_key].append(inst)

    grouped_instances = list(grouped_instances.values())
    rng.shuffle(grouped_instances)
    target_train_size = int(len(instances) * ratio)

    train_instances = []
    test_instances = []
    current_train_size = 0
    for group in grouped_instances:
        group_size = len(group)
        if current_train_size >= target_train_size:
            test_instances.extend(group)
            continue

        deficit = target_train_size - current_train_size
        overshoot = current_train_size + group_size - target_train_size
        should_add_to_train = group_size <= deficit or deficit >= overshoot
        if should_add_to_train:
            train_instances.extend(group)
            current_train_size += group_size
        else:
            test_instances.extend(group)

    rng.shuffle(train_instances)
    rng.shuffle(test_instances)
    return train_instances, test_instances


def split_instances_by_grouped_label_ratios(instances, normal_ratio, anomaly_ratio, rng):
    normal_instances = [inst for inst in instances if inst.label == 'Normal']
    anomaly_instances = [inst for inst in instances if inst.label == 'Anomalous']

    normal_train, normal_test = split_instances_by_sequence_groups(normal_instances, normal_ratio, rng)
    anomaly_train, anomaly_test = split_instances_by_sequence_groups(anomaly_instances, anomaly_ratio, rng)

    train_instances = normal_train + anomaly_train
    test_instances = normal_test + anomaly_test
    rng.shuffle(train_instances)
    rng.shuffle(test_instances)
    return train_instances, test_instances


def collect_event_ids(instances):
    event_ids = set()
    for inst in instances:
        event_ids.update(inst.sequence)
    return event_ids


def count_exact_sequence_overlap(train_instances, test_instances):
    train_sequences = set(tuple(inst.sequence) for inst in train_instances)
    return sum(1 for inst in test_instances if tuple(inst.sequence) in train_sequences)


def get_effective_training_label(instance):
    predicted_label = getattr(instance, 'predicted', '')
    return predicted_label if predicted_label else instance.label


def label_summary(instances, use_training_labels=False):
    if use_training_labels:
        counter = Counter(get_effective_training_label(inst) for inst in instances)
    else:
        counter = Counter(inst.label for inst in instances)
    return '%d Normal / %d Anomalous' % (counter['Normal'], counter['Anomalous'])


def split_has_both_labels(instances):
    labels = {inst.label for inst in instances}
    return 'Normal' in labels and 'Anomalous' in labels


def filter_trainable_parameters(parameters):
    return [parameter for parameter in parameters if parameter.requires_grad]


def set_parameter_trainability(parameters, requires_grad):
    for parameter in parameters:
        parameter.requires_grad = requires_grad


def build_training_tinsts(batch_insts, vocab):
    max_tokens = 500
    max_sequence_length = max(min(len(inst.sequence), max_tokens) for inst in batch_insts)
    tinst = TInstWithLogits(len(batch_insts), max_sequence_length, 2)
    for batch_index, inst in enumerate(batch_insts):
        tinst.src_ids.append(str(inst.id))
        label_id = vocab.tag2id(get_effective_training_label(inst))
        confidence = min(max(0.5 * getattr(inst, 'confidence', 0.0), 0.0), 1.0)
        tinst.tags[batch_index, label_id] = 1.0 - confidence
        tinst.tags[batch_index, 1 - label_id] = confidence
        tinst.g_truth[batch_index] = label_id
        effective_length = min(len(inst.sequence), max_tokens)
        tinst.word_len[batch_index] = effective_length
        for token_index, event_id in enumerate(inst.sequence[:max_tokens]):
            tinst.src_words[batch_index, token_index] = vocab.word2id(event_id)
            tinst.src_masks[batch_index, token_index] = 1
    return tinst


def move_tinst_to_runtime_device(tinst):
    if torch.cuda.is_available():
        tinst.to_cuda(device)
    return tinst


def iterate_batches(instances, batch_size, rng, shuffle=True):
    if not instances:
        return
    indices = np.arange(len(instances))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield [instances[index] for index in batch_indices]


class ReplacementBatchSampler:
    def __init__(self, instances, positive_fraction=0.5, seed_value=seed):
        self.positive_fraction = positive_fraction
        self.rng = np.random.RandomState(seed_value)
        self.normal_instances = [inst for inst in instances if get_effective_training_label(inst) == 'Normal']
        self.anomaly_instances = [inst for inst in instances if get_effective_training_label(inst) == 'Anomalous']
        self.all_instances = list(instances)
        if not self.all_instances:
            raise ValueError('ReplacementBatchSampler requires at least one instance.')

    def sample(self, batch_size):
        if not self.normal_instances or not self.anomaly_instances:
            indices = self.rng.randint(0, len(self.all_instances), size=batch_size)
            return [self.all_instances[index] for index in indices]

        anomaly_batch_size = int(round(batch_size * self.positive_fraction))
        anomaly_batch_size = max(1, min(batch_size - 1, anomaly_batch_size))
        normal_batch_size = batch_size - anomaly_batch_size

        normal_indices = self.rng.randint(0, len(self.normal_instances), size=normal_batch_size)
        anomaly_indices = self.rng.randint(0, len(self.anomaly_instances), size=anomaly_batch_size)

        batch = [self.normal_instances[index] for index in normal_indices]
        batch.extend(self.anomaly_instances[index] for index in anomaly_indices)
        self.rng.shuffle(batch)
        return batch


class MetaLog:
    _logger = logging.getLogger('MetaLog')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'MetaLog.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for MetaLog succeeded, current working directory: %s, logs will be written in %s'
        % (os.getcwd(), LOG_ROOT)
    )

    @property
    def logger(self):
        return MetaLog._logger

    def __init__(self, vocab, num_layer, hidden_size, label2id, backbone='bimamba', dropout=0.0, mamba_state=64,
                 mamba_conv=4, mamba_expand=2, mamba_variant='auto', use_moe=False, moe_num_experts=4,
                 moe_top_k=2, moe_bottleneck_dim=None, moe_temperature=1.5, moe_gate_dropout=0.1,
                 moe_balance_loss_weight=1e-2, moe_diversity_loss_weight=1e-3, moe_z_loss_weight=0.0,
                 use_normality_anchor=True, prototype_scale=1.0, prototype_loss_weight=0.1,
                 prototype_sep_weight=1e-3, prototype_margin_global=1.0, prototype_margin_expert=1.0,
                 prototype_target_normal_only=True, router_use_distance=True):
        self.label2id = label2id
        self.vocab = vocab
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.backbone = backbone.lower()
        self.dropout = dropout
        self.mamba_state = mamba_state
        self.mamba_conv = mamba_conv
        self.mamba_expand = mamba_expand
        self.mamba_variant = mamba_variant
        self.use_moe = use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_bottleneck_dim = moe_bottleneck_dim
        self.moe_temperature = moe_temperature
        self.moe_gate_dropout = moe_gate_dropout
        self.moe_balance_loss_weight = moe_balance_loss_weight
        self.moe_diversity_loss_weight = moe_diversity_loss_weight
        self.moe_z_loss_weight = moe_z_loss_weight
        self.use_normality_anchor = use_normality_anchor
        self.prototype_scale = prototype_scale
        self.prototype_loss_weight = prototype_loss_weight
        self.prototype_sep_weight = prototype_sep_weight
        self.prototype_margin_global = prototype_margin_global
        self.prototype_margin_expert = prototype_margin_expert
        self.prototype_target_normal_only = prototype_target_normal_only
        self.router_use_distance = router_use_distance
        self.model = self._build_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda(device)
        self.loss = nn.BCELoss()
        self.last_metrics = {}

    def _build_model(self):
        if self.backbone != 'bimamba':
            raise ValueError('Unsupported backbone: %s. Only bimamba is kept in the main pipeline.' % self.backbone)
        return AttBiMambaModel(
            self.vocab,
            self.num_layer,
            self.hidden_size,
            dropout=self.dropout,
            mamba_state=self.mamba_state,
            mamba_conv=self.mamba_conv,
            mamba_expand=self.mamba_expand,
            mamba_variant=self.mamba_variant,
            use_moe=self.use_moe,
            moe_num_experts=self.moe_num_experts,
            moe_top_k=self.moe_top_k,
            moe_bottleneck_dim=self.moe_bottleneck_dim,
            moe_temperature=self.moe_temperature,
            moe_gate_dropout=self.moe_gate_dropout,
            moe_balance_loss_weight=self.moe_balance_loss_weight,
            moe_diversity_loss_weight=self.moe_diversity_loss_weight,
            moe_z_loss_weight=self.moe_z_loss_weight,
            use_normality_anchor=self.use_normality_anchor,
            prototype_scale=self.prototype_scale,
            prototype_margin_global=self.prototype_margin_global,
            prototype_margin_expert=self.prototype_margin_expert,
            router_use_distance=self.router_use_distance,
        )

    def _scalarize_metrics(self, metrics):
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    scalar_metrics[key] = float(value.detach().cpu().item())
            elif isinstance(value, (float, int)):
                scalar_metrics[key] = float(value)
        return scalar_metrics

    def _auxiliary_loss(self, logits):
        if hasattr(self.model, 'get_auxiliary_loss'):
            return self.model.get_auxiliary_loss()
        return logits.new_zeros(())

    def _prototype_loss(self, labels, batch_slice=None, normal_only=False):
        if hasattr(self.model, 'get_prototype_loss'):
            loss = self.model.get_prototype_loss(
                labels,
                self.label2id['Anomalous'],
                batch_slice=batch_slice,
                normal_only=normal_only,
            )
            return loss, self._prototype_metrics()
        return labels.float().new_zeros(()), {}

    def _prototype_separation_loss(self):
        if hasattr(self.model, 'get_prototype_separation_loss'):
            return self.model.get_prototype_separation_loss()
        return torch.zeros((), device=device)

    def _prototype_metrics(self):
        if hasattr(self.model, 'get_prototype_metrics'):
            return self._scalarize_metrics(self.model.get_prototype_metrics())
        return {}

    def classification_loss(self, logits, targets):
        tag_probs = sanitize_probs(logits)
        return self.loss(tag_probs, targets)

    def compute_single_batch_loss(self, batch_insts, normal_only_prototype=False):
        tinst = build_training_tinsts(batch_insts, self.vocab)
        move_tinst_to_runtime_device(tinst)
        logits = self.model(tinst.inputs)
        cls_loss = self.classification_loss(logits, tinst.targets)
        aux_loss = self._auxiliary_loss(logits)
        proto_loss, proto_metrics = self._prototype_loss(
            tinst.truth,
            normal_only=normal_only_prototype,
        )
        proto_sep_loss = self._prototype_separation_loss()
        total_loss = (
            cls_loss +
            aux_loss +
            self.prototype_loss_weight * proto_loss +
            self.prototype_sep_weight * proto_sep_loss
        )
        metrics = {
            'cls_loss': float(cls_loss.detach().cpu().item()),
            'aux_loss': float(aux_loss.detach().cpu().item()),
            'proto_loss': float(proto_loss.detach().cpu().item()),
            'proto_sep_loss': float(proto_sep_loss.detach().cpu().item()),
            'total_loss': float(total_loss.detach().cpu().item()),
        }
        if hasattr(self.model, 'get_moe_metrics'):
            metrics.update(self._scalarize_metrics(self.model.get_moe_metrics()))
        metrics.update(proto_metrics)
        self.last_metrics = metrics
        return total_loss, metrics

    def compute_joint_batch_loss(self, source_batch, target_batch, target_weight):
        tinst = build_training_tinsts(source_batch + target_batch, self.vocab)
        move_tinst_to_runtime_device(tinst)
        logits = self.model(tinst.inputs)
        source_batch_size = len(source_batch)
        source_logits = logits[:source_batch_size]
        target_logits = logits[source_batch_size:]
        source_targets = tinst.targets[:source_batch_size]
        target_targets = tinst.targets[source_batch_size:]
        source_labels = tinst.truth[:source_batch_size]
        target_labels = tinst.truth[source_batch_size:]

        source_loss = self.classification_loss(source_logits, source_targets)
        target_loss = self.classification_loss(target_logits, target_targets)
        aux_loss = self._auxiliary_loss(logits)
        source_proto_loss, source_proto_metrics = self._prototype_loss(
            source_labels,
            batch_slice=slice(0, source_batch_size),
            normal_only=False,
        )
        target_proto_loss, target_proto_metrics = self._prototype_loss(
            target_labels,
            batch_slice=slice(source_batch_size, None),
            normal_only=self.prototype_target_normal_only,
        )
        proto_sep_loss = self._prototype_separation_loss()
        total_loss = (
            source_loss +
            target_weight * target_loss +
            aux_loss +
            self.prototype_loss_weight * (source_proto_loss + target_proto_loss) +
            self.prototype_sep_weight * proto_sep_loss
        )

        metrics = {
            'source_cls_loss': float(source_loss.detach().cpu().item()),
            'target_cls_loss': float(target_loss.detach().cpu().item()),
            'aux_loss': float(aux_loss.detach().cpu().item()),
            'source_proto_loss': float(source_proto_loss.detach().cpu().item()),
            'target_proto_loss': float(target_proto_loss.detach().cpu().item()),
            'proto_sep_loss': float(proto_sep_loss.detach().cpu().item()),
            'total_loss': float(total_loss.detach().cpu().item()),
        }
        if hasattr(self.model, 'get_moe_metrics'):
            metrics.update(self._scalarize_metrics(self.model.get_moe_metrics()))
        metrics.update({'source_' + key: value for key, value in source_proto_metrics.items()})
        metrics.update({'target_' + key: value for key, value in target_proto_metrics.items()})
        self.last_metrics = metrics
        return total_loss, metrics

    def predict(self, inputs, threshold=None):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            tag_logits = sanitize_probs(tag_logits)
        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            anomaly_id = self.label2id['Anomalous']
            pred_tags = np.zeros(probs.shape[0])
            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id
        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu()
        return pred_tags, tag_logits

    def collect_anomaly_scores(self, instances, vocab):
        anomaly_id = self.label2id['Anomalous']
        anomaly_scores = []
        gold_labels = []
        with torch.no_grad():
            self.model.eval()
            local_rng = np.random.RandomState(seed)
            for batch in iterate_batches(instances, 1024, local_rng, shuffle=False):
                tinst = build_training_tinsts(batch, vocab)
                move_tinst_to_runtime_device(tinst)
                tag_probs = sanitize_probs(self.model(tinst.inputs))
                anomaly_scores.extend(tag_probs[:, anomaly_id].detach().cpu().numpy().tolist())
                gold_labels.extend(self.label2id[inst.label] for inst in batch)
        return np.asarray(anomaly_scores, dtype=np.float64), np.asarray(gold_labels, dtype=np.int64)

    def _binary_metrics_from_scores(self, gold_labels, anomaly_scores, threshold):
        anomaly_id = self.label2id['Anomalous']
        gold_positive = gold_labels == anomaly_id
        pred_positive = anomaly_scores >= threshold

        TP = int(np.sum(pred_positive & gold_positive))
        TN = int(np.sum((~pred_positive) & (~gold_positive)))
        FP = int(np.sum(pred_positive & (~gold_positive)))
        FN = int(np.sum((~pred_positive) & gold_positive))

        precision = 100 * TP / (TP + FP) if TP + FP else 0
        recall = 100 * TP / (TP + FN) if TP + FN else 0
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0
        fpr = 100 * FP / (FP + TN) if FP + TN else 0

        return {
            'threshold': threshold,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'precision': precision,
            'recall': recall,
            'f': f,
            'fpr': fpr,
        }

    def _ranking_metrics_from_scores(self, gold_labels, anomaly_scores):
        ranking_metrics = {
            'auroc': float('nan'),
            'aucpr': float('nan'),
        }
        if roc_auc_score is None or average_precision_score is None:
            return ranking_metrics
        try:
            ranking_metrics['auroc'] = 100 * roc_auc_score(gold_labels, anomaly_scores)
        except ValueError:
            pass
        try:
            ranking_metrics['aucpr'] = 100 * average_precision_score(gold_labels, anomaly_scores)
        except ValueError:
            pass
        return ranking_metrics

    def evaluate_metrics(self, instances, threshold=0.5, vocab=None):
        if vocab is None:
            vocab = self.vocab
        anomaly_scores, gold_labels = self.collect_anomaly_scores(instances, vocab)
        metrics = self._binary_metrics_from_scores(gold_labels, anomaly_scores, threshold)
        metrics.update(self._ranking_metrics_from_scores(gold_labels, anomaly_scores))
        return metrics

    def tune_threshold(self, instances, vocab, threshold_min=0.1, threshold_max=0.9, threshold_step=0.01,
                       split_name='target-train'):
        if threshold_step <= 0:
            raise ValueError('threshold_step must be positive.')
        thresholds = np.arange(threshold_min, threshold_max + threshold_step * 0.5, threshold_step)
        anomaly_scores, gold_labels = self.collect_anomaly_scores(instances, vocab)

        best_metrics = None
        for threshold in thresholds:
            metrics = self._binary_metrics_from_scores(gold_labels, anomaly_scores, float(threshold))
            if best_metrics is None or metrics['f'] > best_metrics['f'] or (
                metrics['f'] == best_metrics['f'] and metrics['precision'] > best_metrics['precision']
            ):
                best_metrics = metrics
        self.logger.info('Best threshold on %s set = %.3f' % (split_name, best_metrics['threshold']))
        return best_metrics['threshold'], best_metrics

    def load_model_state(self, checkpoint_path, strict=True):
        self.model.load_state_dict(load_checkpoint_state_dict(checkpoint_path), strict=strict)


def build_semantic_encoder(parser_name, dataset, args=None):
    if parser_name != 'parser_free':
        raise ValueError('Unsupported parser: %s. Only parser_free is kept in the main pipeline.' % parser_name)
    cache_dir = get_protocol_option(args, 'plm_cache_dir', PARSER_FREE_DEFAULTS['cache_dir'])
    return ParserFreeEncoder(
        model_name=get_protocol_option(args, 'plm_model', PARSER_FREE_DEFAULTS['model_name']),
        max_length=get_protocol_option(args, 'plm_max_length', PARSER_FREE_DEFAULTS['max_length']),
        batch_size=get_protocol_option(args, 'plm_batch_size', PARSER_FREE_DEFAULTS['batch_size']),
        pooling=get_protocol_option(args, 'plm_pooling', PARSER_FREE_DEFAULTS['pooling']),
        cache_dir=cache_dir if cache_dir else None,
        dataset=dataset,
    )


def prepare_dataset(dataset, parser_name, semantic_encoder):
    processor = Preprocessor()
    instances, _, _ = processor.process(
        dataset=dataset,
        parsing=parser_name,
        cut_func=identity_cut,
        template_encoding=semantic_encoder,
    )
    return processor, instances


def prepare_protocol_context(direction_key, parser_name, protocol='clean', args=None):
    direction = DIRECTION_CONFIGS[direction_key]
    source_semantic_encoder = build_semantic_encoder(parser_name, direction.source_dataset, args)
    target_semantic_encoder = build_semantic_encoder(parser_name, direction.target_dataset, args)

    source_processor, source_instances = prepare_dataset(
        direction.source_dataset,
        parser_name,
        source_semantic_encoder,
    )
    target_processor, target_instances = prepare_dataset(
        direction.target_dataset,
        parser_name,
        target_semantic_encoder,
    )

    rng = np.random.RandomState(seed)
    if protocol == 'clean':
        clean_config = CLEAN_DIRECTION_CONFIGS[direction_key]
        source_train_raw, _ = split_instances_by_ratio(source_instances, clean_config['source_ratio'], rng)
        target_train_raw, target_test_raw = split_instances_by_grouped_label_ratios(
            target_instances,
            clean_config['target_normal_ratio'],
            clean_config['target_anomaly_ratio'],
            rng,
        )
        source_dev_raw = []
        target_dev_raw = []
        selection_domain = 'target'
        selection_split_name = 'target-train'
        target_training_mode = 'gold'
    else:
        raise ValueError('Unknown protocol: %s' % protocol)

    # Always build the merged embedding table from the full source/target event inventories.
    # This removes train-vs-total vocabulary drift and avoids target-test OOV fallbacks.
    source_embeddings = dict(source_processor.embedding)
    target_embeddings = dict(target_processor.embedding)
    merged_embeddings, domain_mappings = build_merged_embeddings({
        direction.source_dataset: source_embeddings,
        direction.target_dataset: target_embeddings,
    })

    source_train = remap_instances(source_train_raw, domain_mappings[direction.source_dataset])
    source_dev = remap_instances(source_dev_raw, domain_mappings[direction.source_dataset])
    target_train = remap_instances(target_train_raw, domain_mappings[direction.target_dataset])
    target_dev = remap_instances(target_dev_raw, domain_mappings[direction.target_dataset])
    target_test = remap_instances(target_test_raw, domain_mappings[direction.target_dataset])
    exact_overlap = count_exact_sequence_overlap(target_train, target_test)
    exact_dev_overlap = count_exact_sequence_overlap(target_train, target_dev)
    selection_split = target_train if selection_domain == 'target' else (source_dev if source_dev else source_train)
    warmup_select_by = 'selection_f1' if selection_domain == 'source' else 'loss'

    vocab = Vocab()
    vocab.load_from_dict(merged_embeddings)

    return {
        'direction': direction,
        'protocol': protocol,
        'vocab': vocab,
        'label2id': target_processor.label2id,
        'source_train': source_train,
        'source_dev': source_dev,
        'target_train': target_train,
        'target_dev': target_dev,
        'target_test': target_test,
        'selection_split': selection_split,
        'selection_split_name': selection_split_name,
        'uses_target_training': bool(target_train),
        'target_training_mode': target_training_mode,
        'warmup_select_by': warmup_select_by,
        'source_embedding_count': len(source_embeddings),
        'target_embedding_count': len(target_embeddings),
        'exact_target_dev_overlap': exact_dev_overlap,
        'exact_target_overlap': exact_overlap,
        'target_dev_oov_events': 0,
        'target_test_oov_events': 0,
        'source_persistence_suffix': getattr(source_semantic_encoder, 'persistence_suffix', ''),
        'target_persistence_suffix': getattr(target_semantic_encoder, 'persistence_suffix', ''),
    }


def partial_load_state_dict(model, checkpoint_path, logger):
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    load_result = model.load_state_dict(state_dict, strict=False)
    logger.info(
        'Loaded warmup checkpoint from %s with %d missing keys and %d unexpected keys.'
        % (checkpoint_path, len(load_result.missing_keys), len(load_result.unexpected_keys))
    )


def save_model_state(model, checkpoint_path):
    output_dir = os.path.dirname(checkpoint_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    state_dict = model.state_dict()
    portable_state_dict = type(state_dict)(
        (key, value.detach().cpu()) for key, value in state_dict.items()
    )
    torch.save(portable_state_dict, checkpoint_path)


def save_threshold(threshold_path, threshold):
    with open(threshold_path, 'w', encoding='utf-8') as writer:
        writer.write('%.8f\n' % threshold)


def build_warmup_optimizer(metalog, args):
    return torch.optim.AdamW(
        filter_trainable_parameters(metalog.model.parameters()),
        lr=args.warmup_lr,
        weight_decay=args.weight_decay,
    )


def build_joint_optimizer(metalog, args):
    param_groups = []
    backbone_params = filter_trainable_parameters(metalog.model.backbone_parameters())
    gate_params = filter_trainable_parameters(metalog.model.gate_parameters())
    expert_params = filter_trainable_parameters(metalog.model.expert_parameters())
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.joint_backbone_lr})
    if gate_params:
        param_groups.append({'params': gate_params, 'lr': args.joint_gate_lr})
    if expert_params:
        param_groups.append({'params': expert_params, 'lr': args.joint_expert_lr})
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def build_calibration_optimizer(metalog, args):
    param_groups = []
    gate_params = filter_trainable_parameters(metalog.model.gate_parameters())
    expert_params = filter_trainable_parameters(metalog.model.expert_parameters())
    if gate_params:
        param_groups.append({'params': gate_params, 'lr': args.calibration_gate_lr})
    if expert_params:
        param_groups.append({'params': expert_params, 'lr': args.calibration_expert_lr})
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def log_epoch_summary(logger, phase_name, epoch, metrics):
    message = ', '.join(['%s=%.4f' % (key, value) for key, value in metrics.items()])
    logger.info('%s epoch %d | %s' % (phase_name, epoch, message))


def evaluate_target(metalog, context, args, split_name):
    selection_split = context['selection_split']
    selection_split_name = context['selection_split_name']
    use_auto_threshold = args.auto_threshold
    if use_auto_threshold and not split_has_both_labels(selection_split):
        metalog.logger.info(
            'Selection split %s only has a single label; falling back to fixed threshold %.3f.'
            % (selection_split_name, args.threshold)
        )
        use_auto_threshold = False
    if use_auto_threshold:
        threshold, selection_metrics = metalog.tune_threshold(
            selection_split,
            context['vocab'],
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            split_name='%s-%s' % (split_name, selection_split_name),
        )
    else:
        threshold = args.threshold
        selection_metrics = metalog.evaluate_metrics(selection_split, threshold=threshold, vocab=context['vocab'])
    test_metrics = metalog.evaluate_metrics(context['target_test'], threshold=threshold, vocab=context['vocab'])
    return threshold, selection_metrics, test_metrics


def maybe_record_epoch_metrics(args, direction_name, phase_name, epoch, threshold, selection_metrics, test_metrics,
                               selected_for_best, phase_mean_loss=None):
    if not args.epoch_metrics_file:
        return
    append_epoch_metrics(args.epoch_metrics_file, {
        'direction': direction_name,
        'phase': phase_name,
        'epoch': epoch,
        'phase_mean_loss': '%.6f' % phase_mean_loss if phase_mean_loss is not None else '',
        'selected_threshold': '%.6f' % threshold,
        'selection_f1': '%.6f' % selection_metrics['f'],
        'test_precision': '%.6f' % test_metrics['precision'],
        'test_recall': '%.6f' % test_metrics['recall'],
        'test_f1': '%.6f' % test_metrics['f'],
        'test_auroc': '%.6f' % test_metrics['auroc'] if not np.isnan(test_metrics['auroc']) else '',
        'test_aucpr': '%.6f' % test_metrics['aucpr'] if not np.isnan(test_metrics['aucpr']) else '',
        'selected_for_best': int(selected_for_best),
    })


def run_warmup(context, args, checkpoint_prefix, select_by='loss'):
    metalog = MetaLog(
        context['vocab'],
        num_layer,
        lstm_hiddens,
        context['label2id'],
        backbone=args.backbone,
        dropout=args.dropout,
        mamba_state=args.mamba_state,
        mamba_conv=args.mamba_conv,
        mamba_expand=args.mamba_expand,
        mamba_variant=args.mamba_variant,
        use_moe=args.use_normality_anchor,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_bottleneck_dim=args.moe_bottleneck_dim,
        moe_temperature=args.moe_temperature,
        moe_gate_dropout=args.moe_gate_dropout,
        moe_balance_loss_weight=args.moe_balance_loss_weight,
        moe_diversity_loss_weight=args.moe_diversity_loss_weight,
        moe_z_loss_weight=args.moe_z_loss_weight,
        use_normality_anchor=args.use_normality_anchor,
        prototype_scale=args.prototype_scale,
        prototype_loss_weight=args.prototype_loss_weight,
        prototype_sep_weight=args.prototype_sep_weight,
        prototype_margin_global=args.prototype_margin_global,
        prototype_margin_expert=args.prototype_margin_expert,
        prototype_target_normal_only=args.prototype_target_normal_only,
        router_use_distance=args.router_use_distance,
    )
    optimizer = build_warmup_optimizer(metalog, args)
    best_value = None
    best_checkpoint = checkpoint_prefix + '_phaseA_best.pt'
    last_checkpoint = checkpoint_prefix + '_phaseA_last.pt'

    for epoch in range(args.warmup_epochs):
        metalog.model.train()
        epoch_losses = []
        epoch_rng = np.random.RandomState(seed + epoch)
        for batch in iterate_batches(context['source_train'], args.source_batch_size, epoch_rng, shuffle=True):
            optimizer.zero_grad()
            loss, metrics = metalog.compute_single_batch_loss(batch, normal_only_prototype=False)
            loss.backward()
            optimizer.step()
            epoch_losses.append(metrics['total_loss'])

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        threshold, selection_metrics, test_metrics = evaluate_target(metalog, context, args, 'PhaseA')
        log_epoch_summary(metalog.logger, 'PhaseA', epoch, {
            'mean_loss': mean_loss,
            'selection_f1': selection_metrics['f'],
            'test_f1': test_metrics['f'],
            'threshold': threshold,
        })
        save_model_state(metalog.model, last_checkpoint)
        if select_by == 'loss':
            current_value = mean_loss
            is_best = best_value is None or current_value < best_value
        elif select_by == 'selection_f1':
            current_value = selection_metrics['f']
            is_best = best_value is None or current_value > best_value
        else:
            raise ValueError('Unknown warmup selection rule: %s' % select_by)
        maybe_record_epoch_metrics(
            args,
            context['direction'].name,
            'phase_a',
            epoch,
            threshold,
            selection_metrics,
            test_metrics,
            is_best,
            phase_mean_loss=mean_loss,
        )
        if is_best:
            best_value = current_value
            save_model_state(metalog.model, best_checkpoint)

    return best_checkpoint


def run_joint_finetune(context, args, checkpoint_prefix, warmup_checkpoint):
    metalog = MetaLog(
        context['vocab'],
        num_layer,
        lstm_hiddens,
        context['label2id'],
        backbone=args.backbone,
        dropout=args.dropout,
        mamba_state=args.mamba_state,
        mamba_conv=args.mamba_conv,
        mamba_expand=args.mamba_expand,
        mamba_variant=args.mamba_variant,
        use_moe=True,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_bottleneck_dim=args.moe_bottleneck_dim,
        moe_temperature=args.moe_temperature,
        moe_gate_dropout=args.moe_gate_dropout,
        moe_balance_loss_weight=args.moe_balance_loss_weight,
        moe_diversity_loss_weight=args.moe_diversity_loss_weight,
        moe_z_loss_weight=args.moe_z_loss_weight,
        use_normality_anchor=args.use_normality_anchor,
        prototype_scale=args.prototype_scale,
        prototype_loss_weight=args.prototype_loss_weight,
        prototype_sep_weight=args.prototype_sep_weight,
        prototype_margin_global=args.prototype_margin_global,
        prototype_margin_expert=args.prototype_margin_expert,
        prototype_target_normal_only=args.prototype_target_normal_only,
        router_use_distance=args.router_use_distance,
    )
    partial_load_state_dict(metalog.model, warmup_checkpoint, metalog.logger)
    optimizer = build_joint_optimizer(metalog, args)
    target_sampler = ReplacementBatchSampler(
        context['target_train'],
        positive_fraction=args.target_positive_fraction,
        seed_value=seed + 101,
    )

    best_selection_f = None
    best_threshold = args.threshold
    best_checkpoint = checkpoint_prefix + '_phaseB_best.pt'
    best_threshold_file = checkpoint_prefix + '_phaseB_best.threshold.txt'
    last_checkpoint = checkpoint_prefix + '_phaseB_last.pt'

    for epoch in range(args.joint_epochs):
        metalog.model.train()
        epoch_metrics = []
        epoch_rng = np.random.RandomState(seed + 200 + epoch)
        for source_batch in iterate_batches(context['source_train'], args.source_batch_size, epoch_rng, shuffle=True):
            target_batch = target_sampler.sample(args.target_batch_size)
            optimizer.zero_grad()
            loss, metrics = metalog.compute_joint_batch_loss(source_batch, target_batch, args.lambda_target)
            loss.backward()
            optimizer.step()
            epoch_metrics.append(metrics)

        mean_metrics = {
            key: float(np.mean([metric[key] for metric in epoch_metrics])) for key in epoch_metrics[0].keys()
        } if epoch_metrics else {'total_loss': 0.0}
        threshold, selection_metrics, test_metrics = evaluate_target(metalog, context, args, 'PhaseB')
        mean_metrics['selection_f1'] = selection_metrics['f']
        mean_metrics['test_f1'] = test_metrics['f']
        mean_metrics['threshold'] = threshold
        log_epoch_summary(metalog.logger, 'PhaseB', epoch, mean_metrics)

        save_model_state(metalog.model, last_checkpoint)
        is_best = best_selection_f is None or selection_metrics['f'] > best_selection_f
        maybe_record_epoch_metrics(
            args,
            context['direction'].name,
            'phase_b',
            epoch,
            threshold,
            selection_metrics,
            test_metrics,
            is_best,
        )
        if is_best:
            best_selection_f = selection_metrics['f']
            best_threshold = threshold
            save_model_state(metalog.model, best_checkpoint)
            save_threshold(best_threshold_file, threshold)

    return best_checkpoint, best_threshold, last_checkpoint


def run_calibration(context, args, checkpoint_prefix, joint_checkpoint):
    metalog = MetaLog(
        context['vocab'],
        num_layer,
        lstm_hiddens,
        context['label2id'],
        backbone=args.backbone,
        dropout=args.dropout,
        mamba_state=args.mamba_state,
        mamba_conv=args.mamba_conv,
        mamba_expand=args.mamba_expand,
        mamba_variant=args.mamba_variant,
        use_moe=True,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_bottleneck_dim=args.moe_bottleneck_dim,
        moe_temperature=args.moe_temperature,
        moe_gate_dropout=args.moe_gate_dropout,
        moe_balance_loss_weight=args.calibration_balance_loss_weight,
        moe_diversity_loss_weight=args.calibration_diversity_loss_weight,
        moe_z_loss_weight=args.moe_z_loss_weight,
        use_normality_anchor=args.use_normality_anchor,
        prototype_scale=args.prototype_scale,
        prototype_loss_weight=args.prototype_loss_weight,
        prototype_sep_weight=args.prototype_sep_weight,
        prototype_margin_global=args.prototype_margin_global,
        prototype_margin_expert=args.prototype_margin_expert,
        prototype_target_normal_only=args.prototype_target_normal_only,
        router_use_distance=args.router_use_distance,
    )
    metalog.load_model_state(joint_checkpoint)
    set_parameter_trainability(metalog.model.backbone_parameters(), False)
    optimizer = build_calibration_optimizer(metalog, args)
    target_sampler = ReplacementBatchSampler(
        context['target_train'],
        positive_fraction=args.target_positive_fraction,
        seed_value=seed + 301,
    )

    best_selection_f = None
    best_threshold = args.threshold
    best_checkpoint = checkpoint_prefix + '_phaseC_best.pt'
    best_threshold_file = checkpoint_prefix + '_phaseC_best.threshold.txt'
    last_checkpoint = checkpoint_prefix + '_phaseC_last.pt'
    steps_per_epoch = max(1, int(math.ceil(len(context['target_train']) / float(args.target_batch_size))))

    for epoch in range(args.calibration_epochs):
        metalog.model.train()
        epoch_metrics = []
        for _ in range(steps_per_epoch):
            target_batch = target_sampler.sample(args.target_batch_size)
            optimizer.zero_grad()
            loss, metrics = metalog.compute_single_batch_loss(
                target_batch,
                normal_only_prototype=args.prototype_target_normal_only,
            )
            loss.backward()
            optimizer.step()
            epoch_metrics.append(metrics)

        mean_metrics = {
            key: float(np.mean([metric[key] for metric in epoch_metrics])) for key in epoch_metrics[0].keys()
        } if epoch_metrics else {'total_loss': 0.0}
        threshold, selection_metrics, test_metrics = evaluate_target(metalog, context, args, 'PhaseC')
        mean_metrics['selection_f1'] = selection_metrics['f']
        mean_metrics['test_f1'] = test_metrics['f']
        mean_metrics['threshold'] = threshold
        log_epoch_summary(metalog.logger, 'PhaseC', epoch, mean_metrics)

        save_model_state(metalog.model, last_checkpoint)
        is_best = best_selection_f is None or selection_metrics['f'] > best_selection_f
        maybe_record_epoch_metrics(
            args,
            context['direction'].name,
            'phase_c',
            epoch,
            threshold,
            selection_metrics,
            test_metrics,
            is_best,
        )
        if is_best:
            best_selection_f = selection_metrics['f']
            best_threshold = threshold
            save_model_state(metalog.model, best_checkpoint)
            save_threshold(best_threshold_file, threshold)

    return best_checkpoint, best_threshold


def final_evaluate(context, args, checkpoint_path, threshold, use_moe=True):
    metalog = MetaLog(
        context['vocab'],
        num_layer,
        lstm_hiddens,
        context['label2id'],
        backbone=args.backbone,
        dropout=args.dropout,
        mamba_state=args.mamba_state,
        mamba_conv=args.mamba_conv,
        mamba_expand=args.mamba_expand,
        mamba_variant=args.mamba_variant,
        use_moe=use_moe,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_bottleneck_dim=args.moe_bottleneck_dim,
        moe_temperature=args.moe_temperature,
        moe_gate_dropout=args.moe_gate_dropout,
        moe_balance_loss_weight=args.calibration_balance_loss_weight,
        moe_diversity_loss_weight=args.calibration_diversity_loss_weight,
        moe_z_loss_weight=args.moe_z_loss_weight,
        use_normality_anchor=args.use_normality_anchor,
        prototype_scale=args.prototype_scale,
        prototype_loss_weight=args.prototype_loss_weight,
        prototype_sep_weight=args.prototype_sep_weight,
        prototype_margin_global=args.prototype_margin_global,
        prototype_margin_expert=args.prototype_margin_expert,
        prototype_target_normal_only=args.prototype_target_normal_only,
        router_use_distance=args.router_use_distance,
    )
    metalog.load_model_state(checkpoint_path)
    metrics = metalog.evaluate_metrics(context['target_test'], threshold=threshold, vocab=context['vocab'])
    metalog.logger.info(
        'Final target test metrics | threshold=%.4f precision=%.4f recall=%.4f f1=%.4f auroc=%.4f aucpr=%.4f'
        % (
            threshold,
            metrics['precision'],
            metrics['recall'],
            metrics['f'],
            metrics['auroc'],
            metrics['aucpr'],
        )
    )
    return metrics


def default_run_name(args, direction_name):
    return (
        '%s_parser=%s_protocol=%s_backbone=%s_hidden=%d_moe=e%d_k%d_na=%d'
        % (
            direction_name,
            args.parser,
            args.protocol,
            args.backbone,
            lstm_hiddens,
            args.moe_num_experts,
            min(args.moe_top_k, args.moe_num_experts),
            1 if args.use_normality_anchor else 0,
        )
    )


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--parser', type=str, default='parser_free', choices=['parser_free'],
                        help='Input pipeline to use.')
    parser.add_argument('--protocol', type=str, default='clean',
                        choices=['clean'],
                        help='Data split protocol.')
    parser.add_argument('--plm_model', type=str, default=PARSER_FREE_DEFAULTS['model_name'],
                        help='Hugging Face model name used by parser-free encoding.')
    parser.add_argument('--plm_max_length', type=int, default=PARSER_FREE_DEFAULTS['max_length'],
                        help='Maximum tokenizer length for parser-free log encoding.')
    parser.add_argument('--plm_batch_size', type=int, default=PARSER_FREE_DEFAULTS['batch_size'],
                        help='Batch size used when caching parser-free log embeddings.')
    parser.add_argument('--plm_pooling', type=str, default=PARSER_FREE_DEFAULTS['pooling'],
                        choices=['mean', 'cls'], help='Pooling strategy for parser-free log encoding.')
    parser.add_argument('--plm_cache_dir', type=str, default=PARSER_FREE_DEFAULTS['cache_dir'],
                        help='Optional cache directory for parser-free text embeddings.')
    parser.add_argument('--backbone', type=str, default='bimamba', choices=['bimamba'],
                        help='Backbone used by the main pipeline.')
    parser.add_argument('--mamba_state', type=int, default=mamba_state, help='BiMamba state expansion.')
    parser.add_argument('--mamba_conv', type=int, default=mamba_conv, help='BiMamba convolution width.')
    parser.add_argument('--mamba_expand', type=int, default=mamba_expand, help='BiMamba expansion factor.')
    parser.add_argument('--mamba_variant', type=str, default=mamba_variant, help='auto, mamba, or mamba2')
    parser.add_argument('--dropout', type=float, default=0.0, help='Sequence encoder dropout.')
    parser.add_argument('--source_batch_size', type=int, default=64, help='Source batch size for phase A/B.')
    parser.add_argument('--target_batch_size', type=int, default=16, help='Target batch size for phase B/C.')
    parser.add_argument('--target_positive_fraction', type=float, default=0.5,
                        help='Target batch anomaly fraction for replacement sampling.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Phase A epochs.')
    parser.add_argument('--joint_epochs', type=int, default=10, help='Phase B epochs.')
    parser.add_argument('--calibration_epochs', type=int, default=3, help='Phase C epochs.')
    parser.add_argument('--warmup_lr', type=float, default=3e-4, help='Phase A learning rate.')
    parser.add_argument('--joint_backbone_lr', type=float, default=1e-4, help='Phase B backbone learning rate.')
    parser.add_argument('--joint_gate_lr', type=float, default=5e-4, help='Phase B gate learning rate.')
    parser.add_argument('--joint_expert_lr', type=float, default=5e-4, help='Phase B expert learning rate.')
    parser.add_argument('--calibration_gate_lr', type=float, default=2e-4,
                        help='Phase C gate learning rate.')
    parser.add_argument('--calibration_expert_lr', type=float, default=2e-4,
                        help='Phase C expert/head learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='AdamW weight decay.')
    parser.add_argument('--lambda_target', type=float, default=4.0, help='Target loss weight in phase B.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Fallback anomaly threshold.')
    parser.add_argument('--auto_threshold', action='store_true',
                        help='Tune threshold on target-train before evaluating target-test.')
    parser.add_argument('--threshold_min', type=float, default=0.1, help='Lower bound for threshold sweep.')
    parser.add_argument('--threshold_max', type=float, default=0.9, help='Upper bound for threshold sweep.')
    parser.add_argument('--threshold_step', type=float, default=0.01, help='Step size for threshold sweep.')
    parser.add_argument('--run_name', type=str, default='', help='Optional checkpoint prefix.')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Checkpoint path used when mode=test.')
    parser.add_argument('--epoch_metrics_file', type=str, default='',
                        help='Optional CSV file for per-epoch target-test metrics.')
    parser.add_argument('--moe_num_experts', type=int, default=4, help='Number of latent experts.')
    parser.add_argument('--moe_top_k', type=int, default=2, help='Top-k routed experts.')
    parser.add_argument('--moe_bottleneck_dim', type=int, default=0,
                        help='Expert bottleneck dim; <=0 means sent_dim // 4.')
    parser.add_argument('--moe_temperature', type=float, default=1.5, help='Router temperature.')
    parser.add_argument('--moe_gate_dropout', type=float, default=0.1, help='Router input dropout.')
    parser.add_argument('--moe_balance_loss_weight', type=float, default=1e-2,
                        help='Phase B MoE balance loss weight.')
    parser.add_argument('--moe_diversity_loss_weight', type=float, default=1e-3,
                        help='Phase B MoE diversity loss weight.')
    parser.add_argument('--calibration_balance_loss_weight', type=float, default=5e-3,
                        help='Phase C MoE balance loss weight.')
    parser.add_argument('--calibration_diversity_loss_weight', type=float, default=0.0,
                        help='Phase C MoE diversity loss weight.')
    parser.add_argument('--moe_z_loss_weight', type=float, default=0.0,
                        help='Optional router z-loss weight.')
    parser.add_argument('--use-normality-anchor', dest='use_normality_anchor', action='store_true',
                        help='Enable normality-anchored drift-aware MoE head.')
    parser.add_argument('--no-use-normality-anchor', dest='use_normality_anchor', action='store_false',
                        help='Disable the prototype-anchored MoE enhancements.')
    parser.add_argument('--prototype-scale', type=float, default=1.0,
                        help='Scale applied to prototype-induced anomaly logits.')
    parser.add_argument('--prototype-loss-weight', type=float, default=0.1,
                        help='Weight applied to pull/push prototype losses.')
    parser.add_argument('--prototype-sep-weight', type=float, default=1e-3,
                        help='Weight applied to expert prototype separation regularization.')
    parser.add_argument('--prototype-margin-global', type=float, default=1.0,
                        help='Margin used to push anomalies away from the global prototype.')
    parser.add_argument('--prototype-margin-expert', type=float, default=1.0,
                        help='Margin used to push anomalies away from expert prototypes.')
    parser.add_argument('--prototype-target-normal-only', dest='prototype_target_normal_only', action='store_true',
                        help='Only apply target-domain prototype pull loss to target normals.')
    parser.add_argument('--no-prototype-target-normal-only', dest='prototype_target_normal_only',
                        action='store_false',
                        help='Apply target-domain prototype loss to both target normals and anomalies.')
    parser.add_argument('--router-use-distance', dest='router_use_distance', action='store_true',
                        help='Use prototype distance and feature norm in router inputs.')
    parser.add_argument('--no-router-use-distance', dest='router_use_distance', action='store_false',
                        help='Disable distance-aware router features.')
    parser.set_defaults(
        use_normality_anchor=True,
        prototype_target_normal_only=True,
        router_use_distance=True,
    )
    return parser


def run_direction(direction_key, args):
    args.moe_bottleneck_dim = args.moe_bottleneck_dim if args.moe_bottleneck_dim > 0 else None
    context = prepare_protocol_context(direction_key, args.parser, protocol=args.protocol, args=args)
    log_prefix = args.run_name if args.run_name else default_run_name(args, direction_key)
    output_model_dir = os.path.join(
        PROJECT_ROOT,
        'outputs/models/%s/%s_%s/model' % (
            args.protocol,
            context['direction'].target_dataset,
            args.parser,
        ),
    )
    checkpoint_prefix = os.path.join(output_model_dir, log_prefix)

    logger = MetaLog._logger
    logger.info(
        'Prepared %s | protocol=%s | source %s train=%d (%s) dev=%d (%s) | target %s train=%d (%s) dev=%d (%s) test=%d (%s)'
        % (
            direction_key,
            context['protocol'],
            context['direction'].source_dataset,
            len(context['source_train']),
            label_summary(context['source_train']),
            len(context['source_dev']),
            label_summary(context['source_dev']),
            context['direction'].target_dataset,
            len(context['target_train']),
            label_summary(context['target_train']),
            len(context['target_dev']),
            label_summary(context['target_dev']),
            len(context['target_test']),
            label_summary(context['target_test']),
        )
    )
    logger.info(
        'Selection split=%s size=%d (%s) | target used in training=%s'
        % (
            context['selection_split_name'],
            len(context['selection_split']),
            label_summary(context['selection_split']),
            'yes' if context['uses_target_training'] else 'no',
        )
    )
    logger.info(
        'Target supervision=%s | effective target-train=%s | source embeddings=%d target embeddings=%d'
        % (
            context['target_training_mode'],
            label_summary(context['target_train'], use_training_labels=True),
            context['source_embedding_count'],
            context['target_embedding_count'],
        )
    )
    if args.parser == 'parser_free':
        logger.info(
            'Parser-free persistences | %s=%s | %s=%s'
            % (
                context['direction'].source_dataset,
                context['source_persistence_suffix'],
                context['direction'].target_dataset,
                context['target_persistence_suffix'],
            )
        )
    logger.info(
        'Leakage guard stats | target dev overlap=%d | target test overlap=%d'
        % (
            context['exact_target_dev_overlap'],
            context['exact_target_overlap'],
        )
    )

    if args.mode == 'test':
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            raise ValueError('mode=test requires --checkpoint.')
        threshold = args.threshold
        use_moe = context['uses_target_training'] or args.use_normality_anchor
        if args.auto_threshold:
            evaluator = MetaLog(
                context['vocab'],
                num_layer,
                lstm_hiddens,
                context['label2id'],
                backbone=args.backbone,
                dropout=args.dropout,
                mamba_state=args.mamba_state,
                mamba_conv=args.mamba_conv,
                mamba_expand=args.mamba_expand,
                mamba_variant=args.mamba_variant,
                use_moe=use_moe,
                moe_num_experts=args.moe_num_experts,
                moe_top_k=args.moe_top_k,
                moe_bottleneck_dim=args.moe_bottleneck_dim,
                moe_temperature=args.moe_temperature,
                moe_gate_dropout=args.moe_gate_dropout,
                moe_balance_loss_weight=args.calibration_balance_loss_weight,
                moe_diversity_loss_weight=args.calibration_diversity_loss_weight,
                moe_z_loss_weight=args.moe_z_loss_weight,
                use_normality_anchor=args.use_normality_anchor,
                prototype_scale=args.prototype_scale,
                prototype_loss_weight=args.prototype_loss_weight,
                prototype_sep_weight=args.prototype_sep_weight,
                prototype_margin_global=args.prototype_margin_global,
                prototype_margin_expert=args.prototype_margin_expert,
                prototype_target_normal_only=args.prototype_target_normal_only,
                router_use_distance=args.router_use_distance,
            )
            evaluator.load_model_state(checkpoint_path)
            threshold, _ = evaluator.tune_threshold(
                context['selection_split'],
                context['vocab'],
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                threshold_step=args.threshold_step,
                split_name=context['selection_split_name'],
            )
        final_evaluate(context, args, checkpoint_path, threshold, use_moe=use_moe)
        return

    warmup_checkpoint = run_warmup(context, args, checkpoint_prefix, select_by=context['warmup_select_by'])
    if not context['uses_target_training']:
        threshold = args.threshold
        if args.auto_threshold:
            evaluator = MetaLog(
                context['vocab'],
                num_layer,
                lstm_hiddens,
                context['label2id'],
                backbone=args.backbone,
                dropout=args.dropout,
                mamba_state=args.mamba_state,
                mamba_conv=args.mamba_conv,
                mamba_expand=args.mamba_expand,
                mamba_variant=args.mamba_variant,
                use_moe=args.use_normality_anchor,
                moe_num_experts=args.moe_num_experts,
                moe_top_k=args.moe_top_k,
                moe_bottleneck_dim=args.moe_bottleneck_dim,
                moe_temperature=args.moe_temperature,
                moe_gate_dropout=args.moe_gate_dropout,
                moe_balance_loss_weight=args.moe_balance_loss_weight,
                moe_diversity_loss_weight=args.moe_diversity_loss_weight,
                moe_z_loss_weight=args.moe_z_loss_weight,
                use_normality_anchor=args.use_normality_anchor,
                prototype_scale=args.prototype_scale,
                prototype_loss_weight=args.prototype_loss_weight,
                prototype_sep_weight=args.prototype_sep_weight,
                prototype_margin_global=args.prototype_margin_global,
                prototype_margin_expert=args.prototype_margin_expert,
                prototype_target_normal_only=args.prototype_target_normal_only,
                router_use_distance=args.router_use_distance,
            )
            evaluator.load_model_state(warmup_checkpoint)
            threshold, _ = evaluator.tune_threshold(
                context['selection_split'],
                context['vocab'],
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                threshold_step=args.threshold_step,
                split_name=context['selection_split_name'],
            )
        final_evaluate(context, args, warmup_checkpoint, threshold, use_moe=args.use_normality_anchor)
        return

    _, _, joint_last_checkpoint = run_joint_finetune(context, args, checkpoint_prefix, warmup_checkpoint)
    calibration_checkpoint, calibration_threshold = run_calibration(
        context,
        args,
        checkpoint_prefix,
        joint_last_checkpoint,
    )
    final_evaluate(context, args, calibration_checkpoint, calibration_threshold, use_moe=True)
