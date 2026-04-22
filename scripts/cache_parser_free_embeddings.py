import sys

sys.path.extend([".", ".."])

import argparse

from preprocessing.Preprocess import Preprocessor
from representations.parser_free import ParserFreeEncoder


def identity_cut(instances):
    return instances, [], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['HDFS', 'BGL', 'BGLSample'],
                        help='Dataset to preprocess with the parser-free pipeline.')
    parser.add_argument('--plm_model', type=str, default='bert-base-uncased',
                        help='Hugging Face model name used by parser-free encoding.')
    parser.add_argument('--plm_max_length', type=int, default=64,
                        help='Maximum tokenizer length for parser-free log encoding.')
    parser.add_argument('--plm_batch_size', type=int, default=64,
                        help='Batch size used when caching parser-free log embeddings.')
    parser.add_argument('--plm_pooling', type=str, default='mean', choices=['mean', 'cls'],
                        help='Pooling strategy for parser-free log encoding.')
    parser.add_argument('--plm_cache_dir', type=str, default='',
                        help='Optional cache directory for parser-free text embeddings.')
    args = parser.parse_args()

    encoder = ParserFreeEncoder(
        model_name=args.plm_model,
        max_length=args.plm_max_length,
        batch_size=args.plm_batch_size,
        pooling=args.plm_pooling,
        cache_dir=args.plm_cache_dir if args.plm_cache_dir else None,
        dataset=args.dataset,
    )
    processor = Preprocessor()
    train, dev, test = processor.process(
        dataset=args.dataset,
        parsing='parser_free',
        template_encoding=encoder,
        cut_func=identity_cut,
    )
    print(
        'Cached parser-free embeddings for dataset=%s | events=%d | instances=%d'
        % (args.dataset, len(processor.embedding), len(train) + len(dev) + len(test))
    )


if __name__ == '__main__':
    main()
