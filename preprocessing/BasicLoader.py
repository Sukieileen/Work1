import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
import abc


class BasicDataLoader():
    def __init__(self):
        self.in_file = None
        self.logger = None
        self.block2emb = {}
        self.blocks = []
        self.templates = {}
        self.log2temp = {}
        self.rex = []
        self.remove_cols = []
        self.id2label = {0: 'Normal', 1: 'Anomalous'}
        self.label2id = {'Normal': 0, 'Anomalous': 1}
        self.block_set = set()
        self.block2seqs = {}
        self.block2label = {}
        self.block2eventseq = {}
        self.log2message = {}
        self.id2embed = {}
        self.semantic_repr_func = None

    @abc.abstractmethod
    def _load_raw_log_seqs(self):
        return

    @abc.abstractmethod
    def logger(self):
        return

    @abc.abstractmethod
    def _pre_process(self, line):
        return

    def parse_by_parser_free(self, persistence_folder, normalizer):
        if not callable(normalizer):
            raise ValueError('Parser-free preprocessing requires a callable normalizer.')

        self._restore()
        if not os.path.exists(persistence_folder):
            os.makedirs(persistence_folder)

        log_event_seq_file = os.path.join(persistence_folder, 'log_sequences.txt')
        log_event_mapping_file = os.path.join(persistence_folder, 'log_event_mapping.dict')
        event_text_file = os.path.join(persistence_folder, 'event_texts.txt')
        events_embedding_file = os.path.join(persistence_folder, 'events.vec')
        start_time = time.time()

        has_event_texts = self._check_file_existence_and_contents(event_text_file)
        has_event_mapping = self._check_parsing_persistences(log_event_mapping_file, log_event_seq_file)
        if has_event_texts and has_event_mapping:
            self.logger.info('Start loading previous parser-free persistences.')
            with open(event_text_file, 'r', encoding='utf-8') as reader:
                self._load_templates(reader)
            self.load_parsing_results(log_event_mapping_file, log_event_seq_file)
        else:
            self.logger.info('Parser-free persistences missing, start rebuilding from raw log text.')
            self.log2message = self._load_log_messages()
            event_text2id = {}
            next_event_id = 1
            for log_id in sorted(self.log2message.keys()):
                normalized_text = normalizer(self.log2message[log_id])
                if normalized_text not in event_text2id:
                    event_text2id[normalized_text] = next_event_id
                    self.templates[next_event_id] = normalized_text
                    next_event_id += 1
                self.log2temp[log_id] = event_text2id[normalized_text]

            for block, seq in self.block2seqs.items():
                self.block2eventseq[block] = [self.log2temp[log_id] for log_id in seq]

            with open(event_text_file, 'w', encoding='utf-8') as writer:
                self._save_templates(writer)
            self._record_parsing_results(log_event_mapping_file, log_event_seq_file)

        self._prepare_semantic_embed(events_embedding_file)
        self.logger.info('All parser-free data preparation finished in %.2f' % (time.time() - start_time))

    def load_parsing_results(self, log_template_mapping_file, event_seq_file):
        self.logger.info('Start loading previous parsing results.')
        start = time.time()
        log_template_mapping_reader = open(log_template_mapping_file, 'r', encoding='utf-8')
        event_seq_reader = open(event_seq_file, 'r', encoding='utf-8')
        self._load_log2temp(log_template_mapping_reader)
        self._load_log_event_seqs(event_seq_reader)
        log_template_mapping_reader.close()
        event_seq_reader.close()
        self.logger.info('Finished in %.2f' % (time.time() - start))

    def _restore(self):
        self.block2emb = {}
        self.templates = {}
        self.log2temp = {}
        self.block2eventseq = {}
        self.id2embed = {}
        self.log2message = {}

    def _save_log_event_seqs(self, writer):
        self.logger.info('Start saving log event sequences.')
        for block, event_seq in self.block2eventseq.items():
            event_seq = map(lambda x: str(x), event_seq)
            seq_str = ' '.join(event_seq)
            writer.write(str(block) + ':' + seq_str + '\n')
        self.logger.info('Log event sequences saved.')

    def _load_log_event_seqs(self, reader):
        for line in reader.readlines():
            tokens = line.strip().split(':')
            block = tokens[0]
            seq = tokens[1].split()
            self.block2eventseq[block] = [int(x) for x in seq]
        self.logger.info('Loaded %d blocks' % len(self.block2eventseq))

    def _prepare_semantic_embed(self, semantic_emb_file):
        if self.semantic_repr_func:
            self.id2embed = self.semantic_repr_func(self.templates)
            with open(semantic_emb_file, 'w', encoding='utf-8') as writer:
                for id, embed in self.id2embed.items():
                    writer.write(str(id) + ' ')
                    writer.write(' '.join([str(x) for x in embed.tolist()]) + '\n')
            self.logger.info(
                'Finish calculating semantic representations, please found the vector file at %s' % semantic_emb_file)
        else:
            self.logger.warning(
                'No template encoder. Please be NOTED that this may lead to duplicate full parsing process.')

        pass

    def _check_parsing_persistences(self, log_template_mapping_file, event_seq_file):
        flag = self._check_file_existence_and_contents(
            log_template_mapping_file) and self._check_file_existence_and_contents(event_seq_file)
        return flag

    def _check_file_existence_and_contents(self, file):
        flag = os.path.exists(file) and os.path.getsize(file) != 0
        self.logger.info('checking file %s ... %s' % (file, str(flag)))
        return flag

    def _record_parsing_results(self, log_template_mapping_file, evet_seq_file):
        # Recording parsing results.
        start_time = time.time()
        log_template_mapping_writer = open(log_template_mapping_file, 'w', encoding='utf-8')
        event_seq_writer = open(evet_seq_file, 'w', encoding='utf-8')
        self._save_log2temp(log_template_mapping_writer)
        self._save_log_event_seqs(event_seq_writer)
        log_template_mapping_writer.close()
        event_seq_writer.close()
        self.logger.info('Done in %.2f' % (time.time() - start_time))

    def _load_templates(self, reader):
        for line in reader.readlines():
            tokens = line.strip().split(',')
            id = tokens[0]
            template = ','.join(tokens[1:])
            self.templates[int(id)] = template
        self.logger.info('Loaded %d templates' % len(self.templates))

    def _save_templates(self, writer):
        for id, template in self.templates.items():
            writer.write(','.join([str(id), template]) + '\n')
        self.logger.info('Templates saved.')

    def _load_log2temp(self, reader):
        for line in reader.readlines():
            logid, tempid = line.strip().split(',')
            self.log2temp[int(logid)] = int(tempid)
        self.logger.info('Loaded %d log sequences and their mappings.' % len(self.log2temp))

    def _save_log2temp(self, writer):
        for log_id, temp_id in self.log2temp.items():
            writer.write(str(log_id) + ',' + str(temp_id) + '\n')
        self.logger.info('Log2Temp saved.')

    def _load_semantic_embed(self, reader):
        for line in reader.readlines():
            token = line.split()
            template_id = int(token[0])
            embed = np.asarray(token[1:], dtype=np.float)
            self.id2embed[template_id] = embed
        self.logger.info('Load %d templates with embedding size %d' % (len(self.id2embed), self.id2embed[1].shape[0]))

    def _iter_raw_log_files(self):
        files = [self.in_file]
        if hasattr(self, 'ab_in_file') and self.ab_in_file:
            files.append(self.ab_in_file)
        return files

    def _load_log_messages(self):
        log_messages = {}
        log_id = 0
        for input_file in self._iter_raw_log_files():
            with open(input_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    log_messages[log_id] = self._pre_process(line)
                    log_id += 1
        self.logger.info('Loaded %d raw log messages for parser-free processing.' % len(log_messages))
        return log_messages
