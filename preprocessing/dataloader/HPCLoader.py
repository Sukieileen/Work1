import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
from collections import OrderedDict
from preprocessing.BasicLoader import BasicDataLoader


class HPCLoader(BasicDataLoader):
    def __init__(
        self,
        in_file=None,
        window_size=120,
        dataset_base=os.path.join(PROJECT_ROOT, 'datasets/HPC'),
        semantic_repr_func=None,
        remove_col_count=8,
        group_key_index=3,
        normal_prefix='-',
    ):
        super(HPCLoader, self).__init__()

        self.logger = logging.getLogger('HPCLoader')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

            file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'HPCLoader.log'))
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
        self.window_size = window_size
        self.dataset_base = dataset_base
        self.remove_cols = list(range(remove_col_count))
        self.group_key_index = group_key_index
        self.normal_prefix = normal_prefix
        self._load_raw_log_seqs()
        self.semantic_repr_func = semantic_repr_func

    def logger(self):
        return self.logger

    def _pre_process(self, line):
        tokens = line.strip().split()
        after_process = []
        for idx, token in enumerate(tokens):
            if idx not in self.remove_cols:
                after_process.append(token)
        return ' '.join(after_process)

    def _iter_log_records(self):
        with open(self.in_file, 'r', encoding='utf-8') as reader:
            for idx, line in enumerate(reader):
                stripped = line.strip()
                if not stripped:
                    continue
                tokens = stripped.split()
                if len(tokens) <= self.group_key_index:
                    self.logger.warning('Skip malformed line %d: %s', idx, stripped[:200])
                    continue
                yield idx, stripped, tokens

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
            self.logger.info('Extraction finished successfully.')
            return

        self.logger.info('Start loading HPC log sequences.')
        nodes = OrderedDict()
        for idx, stripped, tokens in tqdm(self._iter_log_records()):
            node = str(tokens[self.group_key_index])
            if node not in nodes.keys():
                nodes[node] = []
            nodes[node].append((idx, stripped))

        pbar = tqdm(total=len(nodes))
        block_idx = 0
        for node, seq in nodes.items():
            if len(seq) < self.window_size:
                self.blocks.append(str(block_idx))
                self.block2seqs[str(block_idx)] = []
                label = 'Normal'
                for idx, line in seq:
                    self.block2seqs[str(block_idx)].append(int(idx))
                    if not line.startswith(self.normal_prefix):
                        label = 'Anomalous'
                self.block2label[str(block_idx)] = label
                block_idx += 1
            else:
                i = 0
                while i < len(seq):
                    self.blocks.append(str(block_idx))
                    self.block2seqs[str(block_idx)] = []
                    label = 'Normal'
                    for idx, line in seq[i:i + self.window_size]:
                        self.block2seqs[str(block_idx)].append(int(idx))
                        if not line.startswith(self.normal_prefix):
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

    def _load_log_messages(self):
        raw_messages_file = os.path.join(self.dataset_base, 'raw_messages.txt')
        if os.path.exists(raw_messages_file):
            log_messages = {}
            with open(raw_messages_file, 'r', encoding='utf-8') as reader:
                for log_id, line in enumerate(reader):
                    log_messages[log_id] = line.strip()
            self.logger.info(
                'Loaded %d preprocessed raw messages for HPC parser-free processing.' % len(log_messages)
            )
            return log_messages
        return super(HPCLoader, self)._load_log_messages()
