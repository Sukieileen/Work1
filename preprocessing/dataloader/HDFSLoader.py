import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
from preprocessing.BasicLoader import BasicDataLoader
import os.path


class HDFSLoader(BasicDataLoader):
    def __init__(self, in_file=None, datasets_base=os.path.join(PROJECT_ROOT, 'datasets/HDFS'),
                 semantic_repr_func=None):
        super(HDFSLoader, self).__init__()

        # Dispose Loggers.
        self.logger = logging.getLogger('HDFSLoader')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

            file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'HDFSLoader.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.info(
                'Construct self.logger success, current working directory: %s, logs will be written in %s' %
                (os.getcwd(), LOG_ROOT))

        self.blk_rex = re.compile(r'blk_[-]{0,1}[0-9]+')
        if not os.path.exists(in_file):
            self.logger.error('Input file not found, please check.')
            exit(1)
        self.in_file = in_file
        self.remove_cols = [0, 1, 2, 3, 4]
        self.dataset_base = datasets_base
        self._load_raw_log_seqs()
        self._load_hdfs_labels()
        self.semantic_repr_func = semantic_repr_func
        pass

    def logger(self):
        return self.logger

    def _pre_process(self, line):
        tokens = line.strip().split()
        after_process = []
        for idx, token in enumerate(tokens):
            if idx not in self.remove_cols:
                after_process.append(token)
        return ' '.join(after_process)

    def _load_raw_log_seqs(self):
        '''
        Load log sequences from raw HDFS log file.
        :return: Update related attributes in current instance.
        '''
        sequence_file = os.path.join(self.dataset_base, 'raw_log_seqs.txt')
        if not os.path.exists(sequence_file):
            self.logger.info('Start extract log sequences from HDFS raw log file.')
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                log_id = 0
                for line in tqdm(reader.readlines()):
                    processed_line = self._pre_process(line)
                    block_ids = set(re.findall(self.blk_rex, processed_line))
                    if len(block_ids) == 0:
                        self.logger.warning('Failed to parse line: %s . Try with raw log message.' % line)
                        block_ids = set(re.findall(self.blk_rex, line))
                        if len(block_ids) == 0:
                            self.logger.error('Failed, please check the raw log file.')
                        else:
                            self.logger.info('Succeed. %d block ids are found.' % len(block_ids))

                    for block_id in block_ids:
                        if block_id not in self.block2seqs.keys():
                            self.blocks.append(block_id)
                            self.block2seqs[block_id] = []
                        self.block2seqs[block_id].append(log_id)

                    log_id += 1
            with open(sequence_file, 'w', encoding='utf-8') as writer:
                for block in self.blocks:
                    writer.write(':'.join([block, ' '.join([str(x) for x in self.block2seqs[block]])]) + '\n')
        else:
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

        self.logger.info('Extraction finished successfully.')

    def _load_hdfs_labels(self):
        with open(os.path.join(PROJECT_ROOT, 'datasets/HDFS/label.txt'), 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                token = line.strip().split(',')
                block = token[0]
                label = self.id2label[int(token[1])]
                self.block2label[block] = label
