import sys

sys.path.extend([".", ".."])

import argparse
import csv
import os

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from CONSTANTS import PROJECT_ROOT
from approaches.MetaLog import (
    MetaLog,
    lstm_hiddens,
    mamba_conv,
    mamba_expand,
    mamba_state,
    mamba_variant,
    num_layer,
)
from preprocessing.Preprocess import Preprocessor
from preprocessing.datacutter.SimpleCutting import cut_by_316_filter, cut_by_415
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from utils.Vocab import Vocab


def build_eval_context(parser_name):
    dataset = 'BGL'
    template_encoder_BGL = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
    processor_BGL = Preprocessor()
    _, dev_BGL, test_BGL = processor_BGL.process(
        dataset=dataset,
        parsing=parser_name,
        cut_func=cut_by_316_filter,
        template_encoding=template_encoder_BGL.present,
    )
    vocab_BGL = Vocab()
    vocab_BGL.load_from_dict(processor_BGL.embedding)

    dataset = 'HDFS'
    template_encoder_HDFS = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
    processor_HDFS = Preprocessor()
    processor_HDFS.process(
        dataset=dataset,
        parsing=parser_name,
        cut_func=cut_by_415,
        template_encoding=template_encoder_HDFS.present,
    )

    merged_embedding = {}
    for key, value in processor_BGL.embedding.items():
        merged_embedding[key] = value
    for key, value in processor_HDFS.embedding.items():
        merged_embedding[key + 432] = value

    vocab = Vocab()
    vocab.load_from_dict(merged_embedding)

    return {
        'dev_BGL': dev_BGL,
        'test_BGL': test_BGL,
        'vocab': vocab,
        'vocab_BGL': vocab_BGL,
        'label2id': processor_BGL.label2id,
    }


def evaluate_backbone(context, backbone_name, checkpoint_path, threshold_min, threshold_max, threshold_step):
    if backbone_name == 'gru':
        metalog = MetaLog(
            context['vocab'],
            num_layer,
            lstm_hiddens,
            context['label2id'],
            backbone='gru',
        )
    elif backbone_name == 'bimamba':
        metalog = MetaLog(
            context['vocab'],
            num_layer,
            lstm_hiddens,
            context['label2id'],
            backbone='bimamba',
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            mamba_variant=mamba_variant,
        )
    else:
        raise ValueError('Unsupported backbone: %s' % backbone_name)

    metalog.model.load_state_dict(torch.load(checkpoint_path))

    tuned_threshold, dev_metrics = metalog.tune_threshold(
        context['dev_BGL'],
        context['vocab_BGL'],
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        split_name='dev',
    )

    anomaly_scores, gold_labels = metalog.collect_anomaly_scores(context['test_BGL'], context['vocab_BGL'])
    test_metrics = metalog._binary_metrics_from_scores(gold_labels, anomaly_scores, tuned_threshold)
    auroc = roc_auc_score(gold_labels, anomaly_scores)
    aucpr = average_precision_score(gold_labels, anomaly_scores)

    result = {
        'backbone': backbone_name,
        'checkpoint': checkpoint_path,
        'selected_threshold': tuned_threshold,
        'dev_f1': dev_metrics['f'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f'],
        'test_fpr': test_metrics['fpr'],
        'test_auroc': auroc * 100,
        'test_aucpr': aucpr * 100,
    }

    del metalog
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_results(output_file, results):
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fieldnames = [
        'backbone',
        'checkpoint',
        'selected_threshold',
        'dev_f1',
        'test_precision',
        'test_recall',
        'test_f1',
        'test_fpr',
        'test_auroc',
        'test_aucpr',
    ]

    with open(output_file, 'w', encoding='utf-8', newline='') as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames, delimiter='\t')
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow({
                key: ('%.6f' % value if isinstance(value, float) else value)
                for key, value in result.items()
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parser', type=str, default='IBM', help='Log parser name.')
    parser.add_argument('--threshold_min', type=float, default=0.5, help='Lower bound for dev threshold sweep.')
    parser.add_argument('--threshold_max', type=float, default=0.95, help='Upper bound for dev threshold sweep.')
    parser.add_argument('--threshold_step', type=float, default=0.005, help='Step size for dev threshold sweep.')
    parser.add_argument(
        '--gru_checkpoint',
        type=str,
        default=os.path.join(
            PROJECT_ROOT,
            'outputs/models/MetaLog/BGL_IBM/model/layer=2_hidden=100_epoch=10_best.pt',
        ),
        help='Checkpoint path for the GRU model.',
    )
    parser.add_argument(
        '--bimamba_checkpoint',
        type=str,
        default=os.path.join(
            PROJECT_ROOT,
            'outputs/models/MetaLog/BGL_IBM/model/backbone=bimamba_layer=2_hidden=100_epoch=10_best.pt',
        ),
        help='Checkpoint path for the BiMamba model.',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=os.path.join(
            PROJECT_ROOT,
            'backbone_compare/backbone_auc_metrics.tsv',
        ),
        help='Where to write the comparison results.',
    )
    args = parser.parse_args()

    context = build_eval_context(args.parser)
    results = [
        evaluate_backbone(
            context,
            'gru',
            args.gru_checkpoint,
            args.threshold_min,
            args.threshold_max,
            args.threshold_step,
        ),
        evaluate_backbone(
            context,
            'bimamba',
            args.bimamba_checkpoint,
            args.threshold_min,
            args.threshold_max,
            args.threshold_step,
        ),
    ]
    write_results(args.output_file, results)

    for result in results:
        print(
            '%s: threshold=%.3f precision=%.4f recall=%.4f f1=%.4f auroc=%.4f aucpr=%.4f'
            % (
                result['backbone'],
                result['selected_threshold'],
                result['test_precision'],
                result['test_recall'],
                result['test_f1'],
                result['test_auroc'],
                result['test_aucpr'],
            )
        )
    print('Saved comparison file to %s' % args.output_file)


if __name__ == '__main__':
    main()
