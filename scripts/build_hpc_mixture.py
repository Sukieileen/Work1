import sys

sys.path.extend([".", ".."])

import argparse
import gzip
import json
import math
import os
import tarfile
from collections import Counter, defaultdict

from CONSTANTS import *


SOURCE_CONFIGS = {
    'BGL': {
        'format': 'plain',
        'path': os.path.join(PROJECT_ROOT, 'datasets/BGL/BGL.log'),
        'remove_col_count': 9,
        'group_key_index': 3,
        'normal_prefix': '-',
    },
    'TDB': {
        'format': 'tar.gz',
        'path': os.path.join(PROJECT_ROOT, 'datasets/TDB/Thunderbird.tar.gz'),
        'member_name': 'Thunderbird.log',
        'remove_col_count': 8,
        'group_key_index': 3,
        'normal_prefix': '-',
    },
    'Liberty': {
        'format': 'gz',
        'path': os.path.join(PROJECT_ROOT, 'datasets/Liberty/liberty2 (1).gz'),
        'remove_col_count': 8,
        'group_key_index': 3,
        'normal_prefix': '-',
    },
}


def _safe_message_tokens(tokens, remove_col_count):
    if len(tokens) <= remove_col_count:
        return ['<empty>']
    return tokens[remove_col_count:]


def iter_source_lines(config):
    file_format = config['format']
    path = config['path']
    if file_format == 'plain':
        with open(path, 'r', encoding='utf-8', errors='replace') as reader:
            for line in reader:
                yield line
        return
    if file_format == 'gz':
        with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as reader:
            for line in reader:
                yield line
        return
    if file_format == 'tar.gz':
        with tarfile.open(path, 'r:gz') as archive:
            member = archive.getmember(config['member_name'])
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError('Failed to extract member %s from %s' % (config['member_name'], path))
            with extracted:
                for raw in extracted:
                    yield raw.decode('utf-8', errors='replace')
        return
    raise ValueError('Unsupported source format: %s' % file_format)


def compute_chunk_quota(total_lines, source_count, chunk_size, anomaly_ratio):
    base_lines = int(math.floor(total_lines / float(source_count)))
    remainder = total_lines - base_lines * source_count
    base_chunk_count = int(math.ceil(base_lines / float(chunk_size)))
    quotas = {}
    for idx, source_name in enumerate(['BGL', 'TDB', 'Liberty']):
        source_lines = base_lines + (1 if idx < remainder else 0)
        source_chunk_count = int(math.ceil(source_lines / float(chunk_size)))
        anomaly_chunks = int(round(source_chunk_count * anomaly_ratio))
        normal_chunks = max(1, source_chunk_count - anomaly_chunks)
        anomaly_chunks = max(1, anomaly_chunks)
        quotas[source_name] = {
            'target_lines': source_lines,
            'target_chunks': source_chunk_count,
            'normal_chunks': normal_chunks,
            'anomalous_chunks': anomaly_chunks,
        }
    return quotas


def collect_selected_chunks(config, source_name, chunk_size, quota):
    target_normal = quota['normal_chunks']
    target_anomalous = quota['anomalous_chunks']
    node_lines = defaultdict(list)
    node_messages = defaultdict(list)
    node_labels = defaultdict(lambda: 'Normal')
    selected_chunks = []
    selected_chunk_counter = Counter()
    line_counter = Counter()
    scanned_lines = 0

    def maybe_emit_chunk(owner):
        label = node_labels[owner]
        if label == 'Normal' and selected_chunk_counter['Normal'] >= target_normal:
            return
        if label == 'Anomalous' and selected_chunk_counter['Anomalous'] >= target_anomalous:
            return
        selected_chunks.append({
            'source': source_name,
            'owner': owner,
            'label': label,
            'lines': list(node_lines[owner]),
            'messages': list(node_messages[owner]),
        })
        selected_chunk_counter[label] += 1

    for raw in iter_source_lines(config):
        stripped = raw.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) <= config['group_key_index']:
            continue
        owner = str(tokens[config['group_key_index']])
        scanned_lines += 1

        node_lines[owner].append(stripped)
        node_messages[owner].append(' '.join(_safe_message_tokens(tokens, config['remove_col_count'])))
        if stripped.startswith(config['normal_prefix']):
            line_counter['Normal'] += 1
        else:
            line_counter['Anomalous'] += 1
            node_labels[owner] = 'Anomalous'

        if len(node_lines[owner]) >= chunk_size:
            maybe_emit_chunk(owner)
            node_lines[owner] = []
            node_messages[owner] = []
            node_labels[owner] = 'Normal'

        if selected_chunk_counter['Normal'] >= target_normal and selected_chunk_counter['Anomalous'] >= target_anomalous:
            break

    if selected_chunk_counter['Normal'] < target_normal or selected_chunk_counter['Anomalous'] < target_anomalous:
        for owner in sorted(node_lines.keys()):
            if not node_lines[owner]:
                continue
            maybe_emit_chunk(owner)
            if selected_chunk_counter['Normal'] >= target_normal and selected_chunk_counter['Anomalous'] >= target_anomalous:
                break

    return {
        'chunks': selected_chunks,
        'selected_chunk_counter': selected_chunk_counter,
        'scanned_lines': scanned_lines,
        'line_counter': line_counter,
    }


def trim_chunks_to_target_lines(chunks, target_lines):
    selected = []
    total_lines = 0
    label_counter = Counter()
    for chunk in chunks:
        if total_lines >= target_lines:
            break
        selected.append(chunk)
        total_lines += len(chunk['lines'])
        label_counter[chunk['label']] += 1
    return selected, total_lines, label_counter


def iter_round_robin_chunks(selected_by_source):
    ordered = {
        source_name: list(chunks)
        for source_name, chunks in selected_by_source.items()
    }
    active = True
    while active:
        active = False
        for source_name in ['BGL', 'TDB', 'Liberty']:
            if ordered[source_name]:
                active = True
                yield ordered[source_name].pop(0)


def write_hpc_dataset(selected_by_source, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in ['HPC.log', 'raw_messages.txt', 'raw_log_seqs.txt', 'label.txt', 'block_sources.txt',
                     'mixture_metadata.json']:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)

    hpc_log_path = os.path.join(output_dir, 'HPC.log')
    raw_messages_path = os.path.join(output_dir, 'raw_messages.txt')
    raw_seqs_path = os.path.join(output_dir, 'raw_log_seqs.txt')
    label_path = os.path.join(output_dir, 'label.txt')
    block_sources_path = os.path.join(output_dir, 'block_sources.txt')
    metadata_path = os.path.join(output_dir, 'mixture_metadata.json')

    total_lines = 0
    total_blocks = 0
    block_label_counter = Counter()
    line_prefix_counter = Counter()
    source_line_counter = Counter()
    source_block_counter = Counter()
    source_chunk_label_counter = Counter()

    with open(hpc_log_path, 'w', encoding='utf-8') as log_writer, \
            open(raw_messages_path, 'w', encoding='utf-8') as message_writer, \
            open(raw_seqs_path, 'w', encoding='utf-8') as seq_writer, \
            open(label_path, 'w', encoding='utf-8') as label_writer, \
            open(block_sources_path, 'w', encoding='utf-8') as source_writer:
        for block_id, chunk in enumerate(iter_round_robin_chunks(selected_by_source)):
            block_indices = []
            block_label_counter[chunk['label']] += 1
            source_block_counter[chunk['source']] += 1
            source_chunk_label_counter[(chunk['source'], chunk['label'])] += 1
            source_line_counter[chunk['source']] += len(chunk['lines'])
            for raw_line, raw_message in zip(chunk['lines'], chunk['messages']):
                log_writer.write(raw_line + '\n')
                message_writer.write(raw_message + '\n')
                block_indices.append(str(total_lines))
                total_lines += 1
                if raw_line.startswith('-'):
                    line_prefix_counter['Normal'] += 1
                else:
                    line_prefix_counter['Anomalous'] += 1
            seq_writer.write('%d:%s\n' % (block_id, ' '.join(block_indices)))
            label_writer.write('%d:%s\n' % (block_id, chunk['label']))
            source_writer.write('%d:%s:%s:%d\n' % (block_id, chunk['source'], chunk['owner'], len(chunk['lines'])))
            total_blocks += 1

    metadata = {
        'total_lines': total_lines,
        'total_blocks': total_blocks,
        'source_line_counter': dict(source_line_counter),
        'source_block_counter': dict(source_block_counter),
        'line_label_counter': dict(line_prefix_counter),
        'chunk_label_counter': dict(block_label_counter),
        'source_chunk_label_counter': {
            '%s::%s' % (source_name, label): count
            for (source_name, label), count in source_chunk_label_counter.items()
        },
    }
    with open(metadata_path, 'w', encoding='utf-8') as writer:
        json.dump(metadata, writer, indent=2, sort_keys=True)

    return {
        'metadata': metadata,
        'paths': {
            'hpc_log_path': hpc_log_path,
            'raw_messages_path': raw_messages_path,
            'raw_seqs_path': raw_seqs_path,
            'label_path': label_path,
            'block_sources_path': block_sources_path,
            'metadata_path': metadata_path,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size', type=int, default=120)
    parser.add_argument('--total_lines', type=int, default=7500000)
    parser.add_argument('--output_dataset', type=str, default='HPC')
    parser.add_argument('--target_chunk_anomaly_ratio', type=float, default=0.424216,
                        help='Target chunk anomaly ratio; default matches full BGL chunk ratio.')
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, 'datasets', args.output_dataset)
    quotas = compute_chunk_quota(
        total_lines=args.total_lines,
        source_count=3,
        chunk_size=args.chunk_size,
        anomaly_ratio=args.target_chunk_anomaly_ratio,
    )

    selected_by_source = {}
    selection_summary = {}
    for source_name in ['BGL', 'TDB', 'Liberty']:
        collected = collect_selected_chunks(
            SOURCE_CONFIGS[source_name],
            source_name,
            args.chunk_size,
            quotas[source_name],
        )
        trimmed, selected_lines, trimmed_counter = trim_chunks_to_target_lines(
            collected['chunks'],
            quotas[source_name]['target_lines'],
        )
        selected_by_source[source_name] = trimmed
        selection_summary[source_name] = {
            'quota': quotas[source_name],
            'scanned_lines': collected['scanned_lines'],
            'pretrim_chunk_counter': dict(collected['selected_chunk_counter']),
            'posttrim_chunk_counter': dict(trimmed_counter),
            'selected_lines': selected_lines,
        }

    written = write_hpc_dataset(selected_by_source, output_dir)
    print(json.dumps({
        'target_chunk_anomaly_ratio': args.target_chunk_anomaly_ratio,
        'selection_summary': selection_summary,
        'dataset_summary': written['metadata'],
        'paths': written['paths'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
