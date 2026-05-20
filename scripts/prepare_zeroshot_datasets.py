import argparse
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METALOG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for path in (METALOG_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from CONSTANTS import PROJECT_ROOT
from approaches.supervised_protocol import build_semantic_encoder, prepare_dataset


def _ensure_openstack_alias():
    source_dir = os.path.join(PROJECT_ROOT, 'datasets', 'openstack')
    target_dir = os.path.join(PROJECT_ROOT, 'datasets', 'OpenStack')
    if os.path.exists(target_dir):
        return target_dir
    if os.path.exists(source_dir):
        os.symlink(source_dir, target_dir)
    return target_dir


def _ensure_spirit_alias():
    source_dir = os.path.join(PROJECT_ROOT, 'datasets', 'spirit')
    target_dir = os.path.join(PROJECT_ROOT, 'datasets', 'SPIRIT')
    if os.path.exists(target_dir):
        return target_dir
    if os.path.exists(source_dir):
        os.symlink(source_dir, target_dir)
    return target_dir


def _remove_cached_extraction(dataset):
    dataset_dir = os.path.join(PROJECT_ROOT, 'datasets', dataset)
    for name in ('raw_log_seqs.txt', 'label.txt', 'raw_messages.txt', 'HPC.log'):
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            os.remove(path)
    for name in (
        os.path.join('inputs', 'parser_free'),
        os.path.join('persistences', 'parser_free'),
    ):
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            shutil.rmtree(path)


def _metadata_path(dataset):
    return os.path.join(PROJECT_ROOT, 'datasets', dataset, 'zero_shot_prep_metadata.json')


def _metadata_matches(dataset, args):
    path = _metadata_path(dataset)
    if not os.path.exists(path):
        return False
    with open(path, 'r', encoding='utf-8') as reader:
        metadata = json.load(reader)
    if dataset == 'SPIRIT':
        return int(metadata.get('spirit_max_lines', 0)) == int(args.spirit_max_lines)
    return True


def _write_dataset_metadata(dataset, args, summary):
    metadata = dict(summary)
    if dataset == 'SPIRIT':
        metadata['spirit_max_lines'] = int(args.spirit_max_lines)
    with open(_metadata_path(dataset), 'w', encoding='utf-8') as writer:
        json.dump(metadata, writer, indent=2)


def _dataset_summary(dataset):
    dataset_dir = os.path.join(PROJECT_ROOT, 'datasets', dataset)
    label_path = os.path.join(dataset_dir, 'label.txt')
    seq_path = os.path.join(dataset_dir, 'raw_log_seqs.txt')
    labels = {'Normal': 0, 'Anomalous': 0}
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                _, label = line.split(':', 1)
                labels[label] = labels.get(label, 0) + 1
    sequence_count = 0
    if os.path.exists(seq_path):
        with open(seq_path, 'r', encoding='utf-8') as reader:
            sequence_count = sum(1 for line in reader if line.strip())
    return {
        'dataset': dataset,
        'dataset_dir': dataset_dir,
        'sequence_count': sequence_count,
        'label_counts': labels,
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare OpenStack/SPIRIT parser-free datasets for zero-shot tests.')
    parser.add_argument('--datasets', nargs='+', default=['OpenStack', 'SPIRIT'],
                        choices=['OpenStack', 'SPIRIT'])
    parser.add_argument('--plm_model', type=str, default='bert-base-uncased')
    parser.add_argument('--plm_max_length', type=int, default=64)
    parser.add_argument('--plm_batch_size', type=int, default=64)
    parser.add_argument('--plm_pooling', type=str, default='mean', choices=['mean', 'cls'])
    parser.add_argument('--plm_cache_dir', type=str, default='')
    parser.add_argument('--spirit_max_lines', type=int, default=4700000,
                        help='Stream at most this many SPIRIT raw log lines; 4.7M is roughly BGL log scale.')
    parser.add_argument('--rebuild_raw', action='store_true',
                        help='Remove cached raw sequence files before rebuilding.')
    parser.add_argument('--summary_file', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'experiments', 'zeroshot', 'dataset_summary.json'))
    args = parser.parse_args()

    _ensure_openstack_alias()
    _ensure_spirit_alias()

    summaries = []
    for dataset in args.datasets:
        if args.rebuild_raw or (dataset == 'SPIRIT' and not _metadata_matches(dataset, args)):
            _remove_cached_extraction(dataset)
        encoder = build_semantic_encoder('parser_free', dataset, args)
        prepare_dataset(dataset, 'parser_free', encoder)
        summary = _dataset_summary(dataset)
        _write_dataset_metadata(dataset, args, summary)
        summaries.append(summary)

    output_dir = os.path.dirname(args.summary_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.summary_file, 'w', encoding='utf-8') as writer:
        json.dump(summaries, writer, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
