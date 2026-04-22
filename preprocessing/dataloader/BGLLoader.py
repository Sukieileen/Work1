import sys

sys.path.extend([".",".."])
from CONSTANTS import *
from collections import OrderedDict
from preprocessing.BasicLoader import BasicDataLoader


class BGLLoader(BasicDataLoader):
    def __init__(self, in_file=None,
                 window_size=120,
                 dataset_base=os.path.join(PROJECT_ROOT, 'datasets/BGL'),
                 semantic_repr_func=None):
        super(BGLLoader, self).__init__()

        # Construct logger.
        self.logger = logging.getLogger('BGLLoader')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

            file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'BGLLoader.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.info(
                'Construct self.logger success, current working directory: %s, logs will be written in %s' %
                (os.getcwd(), LOG_ROOT))

        if not os.path.exists(in_file):
            self.logger.error('Input file not found, please check.')
            exit(1)
        self.in_file = in_file
        self.remove_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.window_size = window_size
        self.dataset_base = dataset_base
        self._load_raw_log_seqs()
        self.semantic_repr_func = semantic_repr_func
        pass

    def logger(self):
        return self.logger

    def _pre_process(self, line):
        tokens = line.strip().split()
        after_process = []
        for id, token in enumerate(tokens):
            if id not in self.remove_cols:
                after_process.append(token)
        return ' '.join(after_process)
        # return re.sub('[\*\.\?\+\$\^\[\]\(\)\{\}\|\\\/]', '', ' '.join(after_process))

    def _load_raw_log_seqs(self):
        sequence_file = os.path.join(self.dataset_base, 'raw_log_seqs.txt')
        label_file = os.path.join(self.dataset_base, 'label.txt')
        if os.path.exists(sequence_file) and os.path.exists(label_file):
            self.logger.info('Start load from previous extraction. File path %s' % sequence_file)
            with open(sequence_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split(':')
                    block = tokens[0]
                    seq = tokens[1].split()
                    if block not in self.block2seqs.keys():
                        self.block2seqs[block] = []
                        self.blocks.append(block)
                    self.block2seqs[block] = [int(x) for x in seq]
            with open(label_file, 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    block_id, label = line.strip().split(':')
                    self.block2label[block_id] = label

        else:
            self.logger.info('Start loading BGL log sequences.')
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                lines = reader.readlines()
                nodes = OrderedDict()
                for idx, line in enumerate(lines):
                    tokens = line.strip().split()
                    node = str(tokens[3])
                    if node not in nodes.keys():
                        nodes[node] = []
                    nodes[node].append((idx, line.strip()))

                pbar = tqdm(total=len(nodes))

                block_idx = 0
                for node, seq in nodes.items():
                    if len(seq) < self.window_size:
                        self.blocks.append(str(block_idx))
                        self.block2seqs[str(block_idx)] = []
                        label = 'Normal'
                        for (idx, line) in seq:
                            self.block2seqs[str(block_idx)].append(int(idx))
                            if not line.startswith('-'):
                                label = 'Anomalous'
                        self.block2label[str(block_idx)] = label
                        block_idx += 1
                    else:
                        i = 0
                        while i < len(seq):
                            self.blocks.append(str(block_idx))
                            self.block2seqs[str(block_idx)] = []
                            label = 'Normal'
                            for (idx, line) in seq[i:i + self.window_size]:
                                self.block2seqs[str(block_idx)].append(int(idx))
                                if not line.startswith('-'):
                                    label = 'Anomalous'
                            self.block2label[str(block_idx)] = label
                            block_idx += 1
                            i += self.window_size

                    pbar.update(1)

                pbar.close()
            with open(sequence_file, 'w', encoding='utf-8') as writer:
                for block in self.blocks:
                    writer.write(':'.join([block, ' '.join([str(x) for x in self.block2seqs[block]])]) + '\n')

            with open(label_file, 'w', encoding='utf-8') as writer:
                for block in self.block2label.keys():
                    writer.write(':'.join([block, self.block2label[block]]) + '\n')

        self.logger.info('Extraction finished successfully.')
        pass

