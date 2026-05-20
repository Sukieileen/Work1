import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METALOG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for path in (METALOG_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from CONSTANTS import PROJECT_ROOT
from approaches.supervised_protocol import (
    MetaLog,
    build_merged_embeddings,
    build_semantic_encoder,
    final_evaluate,
    load_checkpoint_state_dict,
    lstm_hiddens,
    num_layer,
    prepare_dataset,
    prepare_protocol_context,
    remap_instances,
)
from utils.Vocab import Vocab


def _read_threshold(path, fallback):
    if not path:
        return fallback
    with open(path, 'r', encoding='utf-8') as reader:
        return float(reader.readline().strip())


def _append_csv(path, row):
    fieldnames = [
        'train_direction',
        'zero_target',
        'method',
        'precision',
        'recall',
        'f1',
        'threshold',
        'threshold_source',
        'checkpoint',
        'total',
        'normal',
        'anomalous',
        'tp',
        'tn',
        'fp',
        'fn',
        'known_event_count',
        'zero_event_count',
        'zero_instance_count',
    ]
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, 'a', encoding='utf-8', newline='') as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        if write_header:
            csv_writer.writeheader()
        csv_writer.writerow(row)


def _label_counts(instances):
    normal = sum(1 for inst in instances if inst.label == 'Normal')
    anomalous = sum(1 for inst in instances if inst.label == 'Anomalous')
    return normal, anomalous


def _load_zero_dataset(dataset, args):
    encoder = build_semantic_encoder('parser_free', dataset, args)
    processor, instances = prepare_dataset(dataset, 'parser_free', encoder)
    return processor, instances


def _infer_known_prefix(checkpoint_path):
    marker = os.path.join('outputs', 'models')
    if marker not in checkpoint_path:
        return os.path.splitext(checkpoint_path)[0]
    return os.path.splitext(checkpoint_path)[0]


def build_zero_context(known_direction, zero_target, args):
    known_context = prepare_protocol_context(known_direction, 'parser_free', protocol=args.protocol, args=args)
    zero_processor, zero_instances_raw = _load_zero_dataset(zero_target, args)

    known_embeddings = OrderedDict()
    for word_id in known_context['vocab']._id2word[1:]:
        vocab_index = known_context['vocab']._word2id[word_id]
        known_embeddings[word_id] = known_context['vocab'].embeddings[vocab_index]

    zero_embeddings = dict(zero_processor.embedding)
    merged_embeddings, domain_mappings = build_merged_embeddings({
        'known': known_embeddings,
        zero_target: zero_embeddings,
    })
    zero_instances = remap_instances(zero_instances_raw, domain_mappings[zero_target])
    vocab = Vocab()
    vocab.load_from_dict(merged_embeddings)

    zero_normal, zero_anomalous = _label_counts(zero_instances)
    return {
        'direction': known_context['direction'],
        'protocol': known_context['protocol'],
        'vocab': vocab,
        'label2id': zero_processor.label2id,
        'source_train': known_context['source_train'],
        'source_dev': known_context['source_dev'],
        'target_train': [],
        'target_dev': [],
        'target_test': zero_instances,
        'selection_split': known_context['selection_split'],
        'selection_split_name': known_context['selection_split_name'],
        'uses_target_training': False,
        'target_training_mode': 'zero-shot',
        'warmup_select_by': known_context['warmup_select_by'],
        'source_embedding_count': known_context['source_embedding_count'],
        'target_embedding_count': known_context['target_embedding_count'],
        'exact_target_dev_overlap': 0,
        'exact_target_overlap': 0,
        'target_dev_oov_events': 0,
        'target_test_oov_events': 0,
        'source_persistence_suffix': known_context['source_persistence_suffix'],
        'target_persistence_suffix': getattr(build_semantic_encoder('parser_free', zero_target, args), 'persistence_suffix', ''),
        'known_event_count': len(known_embeddings),
        'zero_event_count': len(zero_embeddings),
        'zero_instance_count': len(zero_instances),
        'zero_normal_count': zero_normal,
        'zero_anomalous_count': zero_anomalous,
    }


def load_checkpoint_with_expanded_vocab(metalog, checkpoint_path):
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    model_state = metalog.model.state_dict()
    embed_key = 'word_embed.weight'
    if embed_key in state_dict and embed_key in model_state:
        saved_embedding = state_dict[embed_key]
        current_embedding = model_state[embed_key]
        if current_embedding.shape[0] >= saved_embedding.shape[0] and current_embedding.shape[1] == saved_embedding.shape[1]:
            current_embedding = current_embedding.clone()
            current_embedding[:saved_embedding.shape[0]] = saved_embedding
            state_dict[embed_key] = current_embedding
    load_result = metalog.model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in load_result.unexpected_keys if key != embed_key]
    if unexpected:
        raise ValueError('Unexpected checkpoint keys: %s' % unexpected[:10])
    return load_result


def evaluate_zero_context(context, args, checkpoint_path, threshold):
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
        use_moe=args.use_moe,
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
    load_checkpoint_with_expanded_vocab(metalog, checkpoint_path)
    return metalog.evaluate_metrics(context['target_test'], threshold=threshold, vocab=context['vocab'])


def main():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--checkpoint', required=True)
    base_parser.add_argument('--threshold_file', type=str, default='')
    base_parser.add_argument('--threshold', type=float, default=0.5)
    base_parser.add_argument('--known_direction', '--train_direction', dest='known_direction',
                             choices=['hdfs_to_hpc', 'hdfs_to_hpc_sr065', 'hpc_to_hdfs', 'hdfs30_hpc065_known_mix'],
                             required=True)
    base_parser.add_argument('--zero_target', choices=['OpenStack', 'SPIRIT'], required=True)
    base_parser.add_argument('--method_name', type=str, default='MetaLog')
    base_parser.add_argument('--threshold_source', type=str, default='')
    base_parser.add_argument('--output', type=str,
                             default=os.path.join(PROJECT_ROOT, 'outputs', 'experiments', 'zeroshot',
                                                  'results_zeroshot_raw.csv'))

    from approaches.supervised_protocol import build_arg_parser

    protocol_parser = build_arg_parser()
    parser = argparse.ArgumentParser(parents=[base_parser], conflict_handler='resolve')
    for action in protocol_parser._actions:
        if action.dest == 'help':
            continue
        if action.dest in {item.dest for item in parser._actions}:
            continue
        parser._add_action(action)
    args = parser.parse_args()

    args.moe_bottleneck_dim = args.moe_bottleneck_dim if args.moe_bottleneck_dim > 0 else None
    threshold = _read_threshold(args.threshold_file, args.threshold)
    threshold_source = args.threshold_source
    if not threshold_source:
        if args.known_direction == 'hdfs30_hpc065_known_mix':
            threshold_source = 'known-mix-dev selection'
        else:
            threshold_source = 'HPC selection' if args.known_direction.startswith('hdfs_to_hpc') else 'HDFS selection'

    context = build_zero_context(args.known_direction, args.zero_target, args)
    metrics = evaluate_zero_context(context, args, args.checkpoint, threshold)
    row = {
        'train_direction': args.known_direction,
        'zero_target': args.zero_target,
        'method': args.method_name,
        'precision': '%.6f' % (metrics['precision'] / 100.0),
        'recall': '%.6f' % (metrics['recall'] / 100.0),
        'f1': '%.6f' % (metrics['f'] / 100.0),
        'threshold': '%.8f' % threshold,
        'threshold_source': threshold_source,
        'checkpoint': os.path.abspath(args.checkpoint),
        'total': metrics['TP'] + metrics['TN'] + metrics['FP'] + metrics['FN'],
        'normal': context['zero_normal_count'],
        'anomalous': context['zero_anomalous_count'],
        'tp': metrics['TP'],
        'tn': metrics['TN'],
        'fp': metrics['FP'],
        'fn': metrics['FN'],
        'known_event_count': context['known_event_count'],
        'zero_event_count': context['zero_event_count'],
        'zero_instance_count': context['zero_instance_count'],
    }
    _append_csv(args.output, row)
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
