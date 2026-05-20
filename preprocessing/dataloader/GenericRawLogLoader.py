import gzip
import os
import sys
import tarfile
from collections import OrderedDict

sys.path.extend([".", ".."])

from CONSTANTS import *
from preprocessing.BasicLoader import BasicDataLoader


class GenericRawLogLoader(BasicDataLoader):
    def __init__(
        self,
        in_file=None,
        dataset_base=None,
        semantic_repr_func=None,
        window_size=120,
        remove_col_count=0,
        group_key_index=None,
        normal_prefix='-',
        raw_format='plain',
        member_name=None,
        mode='cached',
        normal_files=None,
        anomalous_files=None,
        openstack_format=False,
        max_lines=0,
    ):
        super(GenericRawLogLoader, self).__init__()
        self.logger = logging.getLogger('GenericRawLogLoader')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
            file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'GenericRawLogLoader.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

        self.in_file = in_file
        self.dataset_base = dataset_base if dataset_base else os.path.dirname(in_file)
        self.semantic_repr_func = semantic_repr_func
        self.window_size = window_size
        self.remove_col_count = remove_col_count
        self.group_key_index = group_key_index
        self.normal_prefix = normal_prefix
        self.raw_format = raw_format
        self.member_name = member_name
        self.mode = mode
        self.normal_files = normal_files or []
        self.anomalous_files = anomalous_files or []
        self.openstack_format = openstack_format
        self.max_lines = max_lines
        self.log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
        self._openstack_headers = None
        self._openstack_regex = None
        self._load_raw_log_seqs()

    def logger(self):
        return self.logger

    def _pre_process(self, line):
        if self.openstack_format:
            return self._pre_process_openstack(line)
        tokens = line.strip().split()
        if len(tokens) <= self.remove_col_count:
            return '<empty>'
        return ' '.join(tokens[self.remove_col_count:])

    def _pre_process_openstack(self, line):
        if self._openstack_regex is None:
            self._openstack_headers, self._openstack_regex = self.generate_logformat_regex(self.log_format)
        match = self._openstack_regex.search(line.strip())
        if match is None:
            tokens = line.strip().split()
            return ' '.join(tokens[6:]) if len(tokens) > 6 else line.strip()
        message = [match.group(header) for header in self._openstack_headers]
        return message[-1] if message else line.strip()

    def generate_logformat_regex(self, logformat):
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for idx, splitter in enumerate(splitters):
            if idx % 2 == 0:
                regex += re.sub(' +', '\\\s+', splitter)
            else:
                header = splitter.strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        return headers, re.compile('^' + regex + '$')

    def _iter_lines(self, path=None, raw_format=None, member_name=None):
        path = path if path else self.in_file
        raw_format = raw_format if raw_format else self.raw_format
        member_name = member_name if member_name else self.member_name
        if raw_format == 'plain':
            with open(path, 'r', encoding='utf-8', errors='replace') as reader:
                for line in reader:
                    yield line
            return
        if raw_format == 'gz':
            with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as reader:
                for line in reader:
                    yield line
            return
        if raw_format == 'tar.gz':
            with tarfile.open(path, 'r:gz') as archive:
                member = archive.getmember(member_name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError('Failed to extract %s from %s' % (member_name, path))
                with extracted:
                    for raw in extracted:
                        yield raw.decode('utf-8', errors='replace')
            return
        raise ValueError('Unsupported raw format: %s' % raw_format)

    def _load_raw_log_seqs(self):
        sequence_file = os.path.join(self.dataset_base, 'raw_log_seqs.txt')
        label_file = os.path.join(self.dataset_base, 'label.txt')
        if os.path.exists(sequence_file) and os.path.exists(label_file):
            self.logger.info('Start load from previous extraction. File path %s' % sequence_file)
            with open(sequence_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    line = line.strip()
                    if not line:
                        continue
                    block, seq = line.split(':', 1)
                    if block not in self.block2seqs:
                        self.block2seqs[block] = []
                        self.blocks.append(block)
                    self.block2seqs[block] = [int(item) for item in seq.split()]
            with open(label_file, 'r', encoding='utf-8') as reader:
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue
                    block_id, label = line.split(':', 1)
                    self.block2label[block_id] = label
            self.logger.info('Extraction finished successfully.')
            return

        if self.mode == 'openstack':
            self._build_openstack_sequences(sequence_file, label_file)
        elif self.mode == 'hpc':
            self._build_hpc_sequences(sequence_file, label_file)
        else:
            raise FileNotFoundError(
                'Missing cached raw sequences for %s. Expected %s and %s.'
                % (self.dataset_base, sequence_file, label_file)
            )
        self.logger.info('Extraction finished successfully.')

    def _build_openstack_sequences(self, sequence_file, label_file):
        self.logger.info('Start building OpenStack-style fixed-window sequences.')
        raw_messages_path = os.path.join(self.dataset_base, 'raw_messages.txt')
        os.makedirs(self.dataset_base, exist_ok=True)
        block_idx = 0
        log_id = 0
        with open(raw_messages_path, 'w', encoding='utf-8') as message_writer:
            for label, files in [('Normal', self.normal_files), ('Anomalous', self.anomalous_files)]:
                current = []
                for filepath in files:
                    if self.raw_format == 'tar.gz':
                        input_path = self.in_file
                        input_format = 'tar.gz'
                        member_name = filepath
                    else:
                        input_path = filepath
                        input_format = 'plain'
                        member_name = None
                    for line in self._iter_lines(input_path, raw_format=input_format, member_name=member_name):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        message_writer.write(self._pre_process(stripped) + '\n')
                        current.append(log_id)
                        log_id += 1
                        if len(current) >= self.window_size:
                            self._add_block(block_idx, current, label)
                            block_idx += 1
                            current = []
                if current:
                    self._add_block(block_idx, current, label)
                    block_idx += 1
        self._save_sequence_files(sequence_file, label_file)

    def _build_hpc_sequences(self, sequence_file, label_file):
        self.logger.info('Start building HPC-style grouped fixed-window sequences.')
        raw_messages_path = os.path.join(self.dataset_base, 'raw_messages.txt')
        hpc_log_path = os.path.join(self.dataset_base, 'HPC.log')
        os.makedirs(self.dataset_base, exist_ok=True)
        node_lines = OrderedDict()
        node_labels = {}
        block_idx = 0
        log_id = 0
        with open(raw_messages_path, 'w', encoding='utf-8') as message_writer, \
                open(hpc_log_path, 'w', encoding='utf-8') as log_writer:
            for raw in self._iter_lines():
                stripped = raw.strip()
                if not stripped:
                    continue
                tokens = stripped.split()
                if self.group_key_index is not None and len(tokens) <= self.group_key_index:
                    continue
                owner = str(tokens[self.group_key_index]) if self.group_key_index is not None else 'all'
                if owner not in node_lines:
                    node_lines[owner] = []
                    node_labels[owner] = 'Normal'
                log_writer.write(stripped + '\n')
                message_writer.write(self._pre_process(stripped) + '\n')
                node_lines[owner].append(log_id)
                if not stripped.startswith(self.normal_prefix):
                    node_labels[owner] = 'Anomalous'
                log_id += 1
                if len(node_lines[owner]) >= self.window_size:
                    self._add_block(block_idx, node_lines[owner], node_labels[owner])
                    block_idx += 1
                    node_lines[owner] = []
                    node_labels[owner] = 'Normal'
                if self.max_lines and log_id >= self.max_lines:
                    break

            for owner, seq in node_lines.items():
                if not seq:
                    continue
                self._add_block(block_idx, seq, node_labels[owner])
                block_idx += 1
        self._save_sequence_files(sequence_file, label_file)

    def _add_block(self, block_idx, sequence, label):
        block = str(block_idx)
        self.blocks.append(block)
        self.block2seqs[block] = list(sequence)
        self.block2label[block] = label

    def _save_sequence_files(self, sequence_file, label_file):
        with open(sequence_file, 'w', encoding='utf-8') as writer:
            for block in self.blocks:
                writer.write('%s:%s\n' % (block, ' '.join(str(item) for item in self.block2seqs[block])))
        with open(label_file, 'w', encoding='utf-8') as writer:
            for block in self.blocks:
                writer.write('%s:%s\n' % (block, self.block2label[block]))

    def _load_log_messages(self):
        raw_messages_file = os.path.join(self.dataset_base, 'raw_messages.txt')
        if os.path.exists(raw_messages_file):
            log_messages = {}
            with open(raw_messages_file, 'r', encoding='utf-8') as reader:
                for log_id, line in enumerate(reader):
                    log_messages[log_id] = line.strip()
            self.logger.info(
                'Loaded %d preprocessed raw messages from %s.' % (len(log_messages), raw_messages_file)
            )
            return log_messages
        return super(GenericRawLogLoader, self)._load_log_messages()
