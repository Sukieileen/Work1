import sys

sys.path.extend([".", ".."])

import argparse
import csv
import math
from dataclasses import dataclass

from CONSTANTS import *
from entities.TensorInstances import TInstWithLogits
from entities.instances import Instance
from models.gru import AttGRUModel
from models.mamba import AttBiMambaModel
from preprocessing.Preprocess import Preprocessor
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
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
    source_ratio: float
    target_normal_ratio: float
    target_anomaly_ratio: float


DIRECTION_CONFIGS = {
    'hdfs_to_bgl': DirectionConfig(
        name='hdfs_to_bgl',
        source_dataset='HDFS',
        target_dataset='BGL',
        source_ratio=0.3,
        target_normal_ratio=0.3,
        target_anomaly_ratio=0.01,
    ),
    'bgl_to_hdfs': DirectionConfig(
        name='bgl_to_hdfs',
        source_dataset='BGL',
        target_dataset='HDFS',
        source_ratio=1.0,
        target_normal_ratio=0.1,
        target_anomaly_ratio=0.01,
    ),
}


def sanitize_probs(tag_logits):
    tag_probs = F.softmax(tag_logits, dim=1)
    tag_probs = torch.nan_to_num(tag_probs, nan=0.5, posinf=1.0, neginf=0.0)
    tag_probs = torch.clamp(tag_probs, min=1e-6, max=1 - 1e-6)
    return tag_probs


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


def remap_instances(instances, mapping, fallback_event_id=None):
    return [
        clone_instance_with_sequence(inst, [mapping.get(event_id, fallback_event_id) for event_id in inst.sequence])
        for inst in instances
    ]


def split_instances_by_ratio(instances, ratio, rng):
    shuffled = list(instances)
    rng.shuffle(shuffled)
    train_size = int(len(shuffled) * ratio)
    return shuffled[:train_size], shuffled[train_size:]


def split_instances_by_label_ratios(instances, normal_ratio, anomaly_ratio, rng):
    normal_instances = [inst for inst in instances if inst.label == 'Normal']
    anomaly_instances = [inst for inst in instances if inst.label == 'Anomalous']
    rng.shuffle(normal_instances)
    rng.shuffle(anomaly_instances)

    normal_train_size = int(len(normal_instances) * normal_ratio)
    anomaly_train_size = int(len(anomaly_instances) * anomaly_ratio)

    train_instances = normal_instances[:normal_train_size] + anomaly_instances[:anomaly_train_size]
    test_instances = normal_instances[normal_train_size:] + anomaly_instances[anomaly_train_size:]
    rng.shuffle(train_instances)
    rng.shuffle(test_instances)
    return train_instances, test_instances


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


def filter_embeddings_by_event_ids(id2embed, event_ids):
    return {
        event_id: id2embed[event_id]
        for event_id in sorted(event_ids)
        if event_id in id2embed
    }


def count_exact_sequence_overlap(train_instances, test_instances):
    train_sequences = set(tuple(inst.sequence) for inst in train_instances)
    return sum(1 for inst in test_instances if tuple(inst.sequence) in train_sequences)


def count_oov_events(instances, oov_event_id):
    return sum(1 for inst in instances for event_id in inst.sequence if event_id == oov_event_id)


def label_summary(instances):
    counter = Counter(inst.label for inst in instances)
    return '%d Normal / %d Anomalous' % (counter['Normal'], counter['Anomalous'])


def filter_trainable_parameters(parameters):
    return [parameter for parameter in parameters if parameter.requires_grad]


def set_parameter_trainability(parameters, requires_grad):
    for parameter in parameters:
        parameter.requires_grad = requires_grad


def build_supervised_tinsts(batch_insts, vocab):
    max_sequence_length = max(len(inst.sequence) for inst in batch_insts)
    tinst = TInstWithLogits(len(batch_insts), max_sequence_length, 2)
    for batch_index, inst in enumerate(batch_insts):
        tinst.src_ids.append(str(inst.id))
        label_id = vocab.tag2id(inst.label)
        tinst.tags[batch_index, label_id] = 1.0
        tinst.g_truth[batch_index] = label_id
        tinst.word_len[batch_index] = len(inst.sequence)
        for token_index, event_id in enumerate(inst.sequence[:500]):
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
        self.normal_instances = [inst for inst in instances if inst.label == 'Normal']
        self.anomaly_instances = [inst for inst in instances if inst.label == 'Anomalous']
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

    def __init__(self, vocab, num_layer, hidden_size, label2id, backbone='gru', dropout=0.0, mamba_state=64,
                 mamba_conv=4, mamba_expand=2, mamba_variant='auto', use_moe=False, moe_num_experts=4,
                 moe_top_k=2, moe_bottleneck_dim=None, moe_temperature=1.5, moe_gate_dropout=0.1,
                 moe_balance_loss_weight=1e-2, moe_diversity_loss_weight=1e-3, moe_z_loss_weight=0.0):
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
        self.model = self._build_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda(device)
        self.loss = nn.BCELoss()
        self.last_metrics = {}

    def _build_model(self):
        if self.backbone == 'gru':
            return AttGRUModel(
                self.vocab,
                self.num_layer,
                self.hidden_size,
                dropout=self.dropout,
                use_moe=self.use_moe,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_bottleneck_dim=self.moe_bottleneck_dim,
                moe_temperature=self.moe_temperature,
                moe_gate_dropout=self.moe_gate_dropout,
                moe_balance_loss_weight=self.moe_balance_loss_weight,
                moe_diversity_loss_weight=self.moe_diversity_loss_weight,
                moe_z_loss_weight=self.moe_z_loss_weight,
            )
        if self.backbone == 'bimamba':
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
            )
        raise ValueError('Unsupported backbone: %s' % self.backbone)

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

    def classification_loss(self, logits, targets):
        tag_probs = sanitize_probs(logits)
        return self.loss(tag_probs, targets)

    def compute_single_batch_loss(self, batch_insts):
        tinst = build_supervised_tinsts(batch_insts, self.vocab)
        move_tinst_to_runtime_device(tinst)
        logits = self.model(tinst.inputs)
        cls_loss = self.classification_loss(logits, tinst.targets)
        aux_loss = self._auxiliary_loss(logits)
        total_loss = cls_loss + aux_loss
        metrics = {
            'cls_loss': float(cls_loss.detach().cpu().item()),
            'aux_loss': float(aux_loss.detach().cpu().item()),
            'total_loss': float(total_loss.detach().cpu().item()),
        }
        if hasattr(self.model, 'get_moe_metrics'):
            metrics.update(self._scalarize_metrics(self.model.get_moe_metrics()))
        self.last_metrics = metrics
        return total_loss, metrics

    def compute_joint_batch_loss(self, source_batch, target_batch, target_weight):
        tinst = build_supervised_tinsts(source_batch + target_batch, self.vocab)
        move_tinst_to_runtime_device(tinst)
        logits = self.model(tinst.inputs)
        source_batch_size = len(source_batch)
        source_logits = logits[:source_batch_size]
        target_logits = logits[source_batch_size:]
        source_targets = tinst.targets[:source_batch_size]
        target_targets = tinst.targets[source_batch_size:]

        source_loss = self.classification_loss(source_logits, source_targets)
        target_loss = self.classification_loss(target_logits, target_targets)
        aux_loss = self._auxiliary_loss(logits)
        total_loss = source_loss + target_weight * target_loss + aux_loss

        metrics = {
            'source_cls_loss': float(source_loss.detach().cpu().item()),
            'target_cls_loss': float(target_loss.detach().cpu().item()),
            'aux_loss': float(aux_loss.detach().cpu().item()),
            'total_loss': float(total_loss.detach().cpu().item()),
        }
        if hasattr(self.model, 'get_moe_metrics'):
            metrics.update(self._scalarize_metrics(self.model.get_moe_metrics()))
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
                tinst = build_supervised_tinsts(batch, vocab)
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


def build_template_encoder(dataset):
    return Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()


def prepare_dataset(dataset, parser_name, template_encoder):
    processor = Preprocessor()
    instances, _, _ = processor.process(
        dataset=dataset,
        parsing=parser_name,
        cut_func=identity_cut,
        template_encoding=template_encoder.present,
    )
    return processor, instances


def prepare_protocol_context(direction_key, parser_name):
    direction = DIRECTION_CONFIGS[direction_key]
    template_encoder = build_template_encoder(direction.source_dataset)

    source_processor, source_instances = prepare_dataset(direction.source_dataset, parser_name, template_encoder)
    target_processor, target_instances = prepare_dataset(direction.target_dataset, parser_name, template_encoder)

    rng = np.random.RandomState(seed)
    source_train_raw, _ = split_instances_by_ratio(source_instances, direction.source_ratio, rng)
    target_train_raw, target_test_raw = split_instances_by_grouped_label_ratios(
        target_instances,
        direction.target_normal_ratio,
        direction.target_anomaly_ratio,
        rng,
    )

    source_embeddings = filter_embeddings_by_event_ids(
        source_processor.embedding,
        collect_event_ids(source_train_raw),
    )
    target_embeddings = filter_embeddings_by_event_ids(
        target_processor.embedding,
        collect_event_ids(target_train_raw),
    )
    merged_embeddings, domain_mappings = build_merged_embeddings({
        direction.source_dataset: source_embeddings,
        direction.target_dataset: target_embeddings,
    })

    target_oov_token = '__%s_target_oov__' % direction.target_dataset.lower()
    source_train = remap_instances(source_train_raw, domain_mappings[direction.source_dataset])
    target_train = remap_instances(target_train_raw, domain_mappings[direction.target_dataset])
    target_test = remap_instances(
        target_test_raw,
        domain_mappings[direction.target_dataset],
        fallback_event_id=target_oov_token,
    )
    exact_overlap = count_exact_sequence_overlap(target_train, target_test)
    target_test_oov_events = count_oov_events(target_test, target_oov_token)

    vocab = Vocab()
    vocab.load_from_dict(merged_embeddings)

    return {
        'direction': direction,
        'vocab': vocab,
        'label2id': target_processor.label2id,
        'source_train': source_train,
        'target_train': target_train,
        'target_test': target_test,
        'exact_target_overlap': exact_overlap,
        'target_test_oov_events': target_test_oov_events,
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
    if args.auto_threshold:
        threshold, selection_metrics = metalog.tune_threshold(
            context['target_train'],
            context['vocab'],
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            split_name='%s-target-train' % split_name,
        )
    else:
        threshold = args.threshold
        selection_metrics = metalog.evaluate_metrics(context['target_train'], threshold=threshold, vocab=context['vocab'])
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


def run_warmup(context, args, checkpoint_prefix):
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
        use_moe=False,
    )
    optimizer = build_warmup_optimizer(metalog, args)
    best_loss = None
    best_checkpoint = checkpoint_prefix + '_phaseA_best.pt'
    last_checkpoint = checkpoint_prefix + '_phaseA_last.pt'

    for epoch in range(args.warmup_epochs):
        metalog.model.train()
        epoch_losses = []
        epoch_rng = np.random.RandomState(seed + epoch)
        for batch in iterate_batches(context['source_train'], args.source_batch_size, epoch_rng, shuffle=True):
            optimizer.zero_grad()
            loss, metrics = metalog.compute_single_batch_loss(batch)
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
        is_best = best_loss is None or mean_loss < best_loss
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
            best_loss = mean_loss
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
            loss, metrics = metalog.compute_single_batch_loss(target_batch)
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


def final_evaluate(context, args, checkpoint_path, threshold):
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
        '%s_backbone=%s_hidden=%d_moe=e%d_k%d'
        % (
            direction_name,
            args.backbone,
            lstm_hiddens,
            args.moe_num_experts,
            min(args.moe_top_k, args.moe_num_experts),
        )
    )


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--parser', type=str, default='IBM', help='Log parser name.')
    parser.add_argument('--backbone', type=str, default='bimamba', help='gru or bimamba')
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
    return parser


def run_direction(direction_key, args):
    args.moe_bottleneck_dim = args.moe_bottleneck_dim if args.moe_bottleneck_dim > 0 else None
    context = prepare_protocol_context(direction_key, args.parser)
    log_prefix = args.run_name if args.run_name else default_run_name(args, direction_key)
    output_model_dir = os.path.join(
        PROJECT_ROOT,
        'outputs/models/protocol_%s/%s_%s/model' % (
            direction_key,
            context['direction'].target_dataset,
            args.parser,
        ),
    )
    checkpoint_prefix = os.path.join(output_model_dir, log_prefix)

    logger = MetaLog._logger
    logger.info(
        'Prepared %s | source %s train=%d (%s) | target %s train=%d (%s) | target test=%d (%s)'
        % (
            direction_key,
            context['direction'].source_dataset,
            len(context['source_train']),
            label_summary(context['source_train']),
            context['direction'].target_dataset,
            len(context['target_train']),
            label_summary(context['target_train']),
            len(context['target_test']),
            label_summary(context['target_test']),
        )
    )
    logger.info(
        'Leakage guard stats | exact target sequence overlap=%d | target test OOV events=%d'
        % (context['exact_target_overlap'], context['target_test_oov_events'])
    )

    if args.mode == 'test':
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            raise ValueError('mode=test requires --checkpoint.')
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
                use_moe=True,
                moe_num_experts=args.moe_num_experts,
                moe_top_k=args.moe_top_k,
                moe_bottleneck_dim=args.moe_bottleneck_dim if args.moe_bottleneck_dim > 0 else None,
                moe_temperature=args.moe_temperature,
                moe_gate_dropout=args.moe_gate_dropout,
                moe_balance_loss_weight=args.calibration_balance_loss_weight,
                moe_diversity_loss_weight=args.calibration_diversity_loss_weight,
                moe_z_loss_weight=args.moe_z_loss_weight,
            )
            evaluator.load_model_state(checkpoint_path)
            threshold, _ = evaluator.tune_threshold(
                context['target_train'],
                context['vocab'],
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                threshold_step=args.threshold_step,
                split_name='target-train',
            )
        final_evaluate(context, args, checkpoint_path, threshold)
        return

    warmup_checkpoint = run_warmup(context, args, checkpoint_prefix)
    _, _, joint_last_checkpoint = run_joint_finetune(context, args, checkpoint_prefix, warmup_checkpoint)
    calibration_checkpoint, calibration_threshold = run_calibration(
        context,
        args,
        checkpoint_prefix,
        joint_last_checkpoint,
    )
    final_evaluate(context, args, calibration_checkpoint, calibration_threshold)
