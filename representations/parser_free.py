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
    VERSION = 'v1'

    def __init__(self, lowercase=True):
        self.lowercase = lowercase
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
        for compiled_re in self.id_res:
            text = compiled_re.sub(' <id> ', text)
        text = self.float_re.sub(' <num> ', text)
        text = self.num_re.sub(' <num> ', text)
        text = self.separator_re.sub(' ', text)
        text = self.space_re.sub(' ', text).strip()
        return text if text else '<empty>'


class ParserFreeEncoder(object):
    def __init__(self, model_name='bert-base-uncased', max_length=64, batch_size=64, pooling='mean',
                 cache_dir=None, lowercase=True):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self.runtime_device = device
        self.logger = _build_logger()
        self.normalizer = LogNormalizer(lowercase=lowercase)
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
