import sys

sys.path.extend([".", ".."])

import hashlib
import json
import pickle

from CONSTANTS import *

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


def _build_logger():
    logger = logging.getLogger('ParserFreeEncoder')
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'ParserFreeEncoder.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info(
        'Construct ParserFreeEncoder logger success, current working directory: %s, logs will be written in %s'
        % (os.getcwd(), LOG_ROOT)
    )
    return logger


class LogNormalizer(object):
    VERSION = 'v4'

    def __init__(self, lowercase=True, dataset=None):
        self.lowercase = lowercase
        self.dataset = dataset.upper() if dataset else ''
        self.uuid_re = re.compile(
            r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
            re.IGNORECASE,
        )
        self.ip_port_re = re.compile(r'(?:(?:\d{1,3}\.){3}\d{1,3})\s*:\s*\d{2,5}')
        self.ip_re = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.port_phrase_re = re.compile(r'\bport\s+\d{2,5}\b')
        self.path_re = re.compile(r'(?:(?:[a-zA-Z]:)?/[\w\.\-]+(?:/[\w\.\-]+)+/?)(?=[\s,;:]|$)')
        self.hex_re = re.compile(r'\b0x[0-9a-f]+\b', re.IGNORECASE)
        self.float_re = re.compile(r'\b[-+]?\d+\.\d+\b')
        self.num_re = re.compile(r'\b[-+]?\d+\b')
        self.id_res = [
            re.compile(r'\bblk_-?\d+\b', re.IGNORECASE),
            re.compile(r'\bjob_\d+(?:_\d+)+\b', re.IGNORECASE),
            re.compile(r'\btask_\d+(?:_\d+|_[a-z])+?\b', re.IGNORECASE),
            re.compile(r'\battempt_\d+(?:_\d+|_[a-z])+?\b', re.IGNORECASE),
            re.compile(r'\bcontainer_[\d_]+\b', re.IGNORECASE),
            re.compile(r'\bapplication_[\d_]+\b', re.IGNORECASE),
            re.compile(r'\bappattempt_[\d_]+\b', re.IGNORECASE),
            re.compile(r'\breq-[0-9a-f\-]+\b', re.IGNORECASE),
        ]
        self.separator_re = re.compile(r'[^\w<>]+')
        self.space_re = re.compile(r'\s+')
        self.bgl_datetime_re = re.compile(
            r'\b(?:mon|tue|wed|thu|fri|sat|sun)\s+'
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+'
            r'\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+[a-z]{3}\s+\d{4}\b',
            re.IGNORECASE,
        )
        self.bgl_colon_hexseq_re = re.compile(r'\b(?:[0-9a-f]{2}:){3,}[0-9a-f]{2}\b', re.IGNORECASE)
        self.bgl_location_re = re.compile(r'\br\d{2}(?:-[a-z0-9]+)+(?::j\d{1,2}-u\d{1,2})?\b', re.IGNORECASE)
        self.bgl_slot_re = re.compile(r'(?<!\w)(?:j|u)\d{1,2}(?!\w)', re.IGNORECASE)
        self.bgl_hex8_re = re.compile(
            r'\b(?=[0-9a-f]{8}\b)(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{8}\b',
            re.IGNORECASE,
        )
        self.bgl_long_hex_re = re.compile(
            r'\b(?=[0-9a-f]{6,}\b)(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{6,}\b',
            re.IGNORECASE,
        )
        self.bgl_assignment_re = re.compile(r'([a-z][a-z0-9_]*)\s*(?:=|:)\s*([0-9a-f]{4,})\b', re.IGNORECASE)
        self.bgl_location_token_re = re.compile(r'^r\d{2}(?:-[a-z0-9]+)+(?::j\d{1,2}-u\d{1,2})?$', re.IGNORECASE)
        self.bgl_slot_token_re = re.compile(r'^(?:j|u)\d{1,2}$', re.IGNORECASE)
        self.bgl_long_hex_token_re = re.compile(r'^[0-9a-f]{5,}$', re.IGNORECASE)
        self.bgl_short_hex_token_re = re.compile(r'^[0-9a-f]{2}$', re.IGNORECASE)

    def normalize(self, text):
        if text is None:
            return '<empty>'

        text = text.strip()
        if not text:
            return '<empty>'

        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        if self.lowercase:
            text = text.lower()

        text = self.uuid_re.sub(' <uuid> ', text)
        text = self.ip_port_re.sub(' <ip> <port> ', text)
        text = self.ip_re.sub(' <ip> ', text)
        text = self.port_phrase_re.sub(' port <port> ', text)
        text = self.path_re.sub(' <path> ', text)
        text = self.hex_re.sub(' <hex> ', text)
        if self.dataset == 'BGL':
            text = self._normalize_bgl(text)
        for compiled_re in self.id_res:
            text = compiled_re.sub(' <id> ', text)
        text = self.float_re.sub(' <num> ', text)
        text = self.num_re.sub(' <num> ', text)
        text = self.separator_re.sub(' ', text)
        text = self.space_re.sub(' ', text).strip()
        return text if text else '<empty>'

    def _normalize_bgl(self, text):
        text = self.bgl_datetime_re.sub(' <datetime> ', text)
        text = self.bgl_colon_hexseq_re.sub(' <hexseq> ', text)
        text = self.bgl_location_re.sub(' <loc> ', text)
        text = self.bgl_slot_re.sub(' <slot> ', text)
        text = self.bgl_hex8_re.sub(lambda match: ' %s ' % self._bucket_bgl_hex(match.group(0)), text)
        text = self.bgl_long_hex_re.sub(' <hex> ', text)
        text = self.bgl_assignment_re.sub(r'\1 <hex>', text)
        tokens = text.split()
        normalized_tokens = []
        token_index = 0
        while token_index < len(tokens):
            token = tokens[token_index]
            lowered_token = token.lower()

            if self.bgl_location_token_re.fullmatch(lowered_token):
                normalized_tokens.append('<loc>')
                token_index += 1
                continue

            if self.bgl_slot_token_re.fullmatch(lowered_token):
                normalized_tokens.append('<slot>')
                token_index += 1
                continue

            hex_run_end = token_index
            saw_alpha_hex = False
            while hex_run_end < len(tokens):
                run_token = tokens[hex_run_end].lower()
                if not self.bgl_short_hex_token_re.fullmatch(run_token):
                    break
                if any(ch in 'abcdef' for ch in run_token):
                    saw_alpha_hex = True
                hex_run_end += 1
            if hex_run_end - token_index >= 4 and saw_alpha_hex:
                normalized_tokens.append('<hexseq>')
                token_index = hex_run_end
                continue

            if self.bgl_long_hex_token_re.fullmatch(lowered_token):
                normalized_tokens.append('<hex>')
                token_index += 1
                continue

            normalized_tokens.append(token)
            token_index += 1

        return ' '.join(normalized_tokens)

    def _bucket_bgl_hex(self, token):
        lowered = token.lower()
        return '<hexp_%s>' % lowered[:4]


class ParserFreeEncoder(object):
    def __init__(self, model_name='bert-base-uncased', max_length=64, batch_size=64, pooling='mean',
                 cache_dir=None, lowercase=True, dataset=None):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self.dataset = dataset.upper() if dataset else ''
        self.runtime_device = device
        self.logger = _build_logger()
        self.normalizer = LogNormalizer(lowercase=lowercase, dataset=self.dataset)
        cache_root = cache_dir if cache_dir else os.path.join(PROJECT_ROOT, 'outputs/parser_free_cache')
        if not os.path.exists(cache_root):
            os.makedirs(cache_root)
        self.cache_dir = cache_root

        self.config_signature = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'pooling': self.pooling,
            'normalizer_version': self.normalizer.VERSION,
            'lowercase': lowercase,
            'dataset': self.dataset,
        }
        signature_json = json.dumps(self.config_signature, sort_keys=True)
        self.persistence_suffix = hashlib.md5(signature_json.encode('utf-8')).hexdigest()[:12]
        model_cache_name = self.model_name.replace('/', '__')
        self.cache_path = os.path.join(self.cache_dir, '%s_%s.pkl' % (model_cache_name, self.persistence_suffix))
        self.embedding_cache = None
        self.tokenizer = None
        self.model = None

    def normalize(self, text):
        return self.normalizer.normalize(text)

    def present(self, id2templates):
        if not id2templates:
            return {}

        self._load_cache()
        ordered_event_ids = sorted(id2templates.keys())
        unique_texts = list(dict.fromkeys(id2templates[event_id] for event_id in ordered_event_ids))
        missing_texts = [text for text in unique_texts if text not in self.embedding_cache]
        if missing_texts:
            self.logger.info(
                'Encoding %d unseen parser-free events with model=%s pooling=%s max_length=%d batch_size=%d'
                % (len(missing_texts), self.model_name, self.pooling, self.max_length, self.batch_size)
            )
            encoded_vectors = self._encode_texts(missing_texts)
            for text, vector in zip(missing_texts, encoded_vectors):
                self.embedding_cache[text] = vector
            self._save_cache()

        return {
            event_id: np.asarray(self.embedding_cache[id2templates[event_id]], dtype=np.float64)
            for event_id in ordered_event_ids
        }

    def _load_cache(self):
        if self.embedding_cache is not None:
            return

        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as reader:
                payload = pickle.load(reader)
            if payload.get('signature') == self.config_signature:
                self.embedding_cache = payload.get('vectors', {})
                self.logger.info(
                    'Loaded parser-free embedding cache with %d entries from %s'
                    % (len(self.embedding_cache), self.cache_path)
                )
                return

        self.embedding_cache = {}

    def _save_cache(self):
        with open(self.cache_path, 'wb') as writer:
            pickle.dump(
                {
                    'signature': self.config_signature,
                    'vectors': self.embedding_cache,
                },
                writer,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def _load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return

        if AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                'Parser-free mode requires transformers. Please install the new parser-free dependencies first.'
            )

        local_files_only = os.environ.get('TRANSFORMERS_OFFLINE', '0').lower() in ('1', 'true', 'yes')
        self.logger.info(
            'Loading parser-free PLM model=%s on device=%s local_files_only=%s'
            % (self.model_name, str(self.runtime_device), str(local_files_only))
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(self.model_name, local_files_only=local_files_only)
        self.model.to(self.runtime_device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _encode_texts(self, texts):
        self._load_model()
        all_vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start:start + self.batch_size]
                encoded_batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                )
                encoded_batch = {
                    key: value.to(self.runtime_device)
                    for key, value in encoded_batch.items()
                }
                outputs = self.model(**encoded_batch)
                hidden_states = outputs.last_hidden_state
                attention_mask = encoded_batch['attention_mask']
                pooled = self._pool(hidden_states, attention_mask)
                all_vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

        if not all_vectors:
            hidden_size = getattr(self.model.config, 'hidden_size', 768)
            return np.zeros((0, hidden_size), dtype=np.float32)
        return np.concatenate(all_vectors, axis=0)

    def _pool(self, hidden_states, attention_mask):
        if self.pooling == 'cls':
            return hidden_states[:, 0, :]

        if self.pooling != 'mean':
            raise ValueError('Unsupported parser-free pooling strategy: %s' % self.pooling)

        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        masked_hidden_states = hidden_states * mask
        token_counts = mask.sum(dim=1).clamp(min=1.0)
        return masked_hidden_states.sum(dim=1) / token_counts
