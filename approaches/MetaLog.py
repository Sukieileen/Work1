import sys
sys.path.extend([".", ".."])
from CONSTANTS import *
from sklearn.decomposition import FastICA
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from representations.sequences.statistics import Sequential_TF
from preprocessing.datacutter.SimpleCutting import cut_by_217, cut_by_316, cut_by_415, cut_by_514, cut_by_316_filter, cut_by_415_filter, cut_by_226_filter, cut_by_514_filter, cut_by_613_filter
from preprocessing.AutoLabeling import Probabilistic_Labeling
from preprocessing.Preprocess import Preprocessor
from module.Optimizer import Optimizer
from module.Common import data_iter, generate_tinsts_binary_label
from models.gru import AttGRUModel
from models.mamba import AttBiMambaModel
from utils.Vocab import Vocab


lstm_hiddens = 100
num_layer = 2
batch_size = 100
epochs = 10
mamba_state = 64
mamba_conv = 4
mamba_expand = 2
mamba_variant = 'auto'


def get_backbone_lr(backbone):
    return 2e-4 if backbone.lower() == 'bimamba' else 2e-3


def sanitize_probs(tag_logits):
    tag_probs = F.softmax(tag_logits, dim=1)
    tag_probs = torch.nan_to_num(tag_probs, nan=0.5, posinf=1.0, neginf=0.0)
    tag_probs = torch.clamp(tag_probs, min=1e-6, max=1 - 1e-6)
    return tag_probs


def get_updated_network(old, new, lr, load=False):
    updated_theta = {}
    state_dicts = old.state_dict()
    param_dicts = dict(old.named_parameters())

    for i, (k, v) in enumerate(state_dicts.items()):
        if k in param_dicts.keys() and param_dicts[k].grad is not None:
            updated_theta[k] = param_dicts[k] - lr * param_dicts[k].grad
        else:
            updated_theta[k] = state_dicts[k]
    if load:
        new.load_state_dict(updated_theta)
    else:
        new = put_theta(new, updated_theta)
    return new


def put_theta(model, theta):
    def k_param_fn(tmp_model, name=None):
        for (k, v) in tmp_model._parameters.items():
            if not isinstance(v, torch.Tensor):
                continue
            full_name = k if name is None else str(name + '.' + k)
            if full_name in theta:
                tmp_model._parameters[k] = theta[full_name]
        for (k, v) in tmp_model._modules.items():
            if name is None:
                k_param_fn(v, name=str(k))
            else:
                k_param_fn(v, name=str(name + '.' + k))

    k_param_fn(model)
    return model


class MetaLog:
    _logger = logging.getLogger('MetaLog')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'MetaLog.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for MetaLog succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return MetaLog._logger

    def __init__(self, vocab, num_layer, hidden_size, label2id, backbone='gru', mamba_state=64,
                 mamba_conv=4, mamba_expand=2, mamba_variant='auto'):
        self.label2id = label2id
        self.vocab = vocab
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.backbone = backbone.lower()
        self.mamba_state = mamba_state
        self.mamba_conv = mamba_conv
        self.mamba_expand = mamba_expand
        self.mamba_variant = mamba_variant
        self.batch_size = 128
        self.test_batch_size = 1024
        self.model = self._build_model()
        self.bk_model = self._build_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda(device)
            self.bk_model = self.bk_model.cuda(device)
        self.loss = nn.BCELoss()

    def _build_model(self):
        if self.backbone == 'gru':
            return AttGRUModel(self.vocab, self.num_layer, self.hidden_size)
        if self.backbone == 'bimamba':
            return AttBiMambaModel(
                self.vocab,
                self.num_layer,
                self.hidden_size,
                mamba_state=self.mamba_state,
                mamba_conv=self.mamba_conv,
                mamba_expand=self.mamba_expand,
                mamba_variant=self.mamba_variant,
            )
        raise ValueError('Unsupported backbone: %s' % self.backbone)

    def forward(self, inputs, targets):
        tag_logits = self.model(inputs)
        tag_probs = sanitize_probs(tag_logits)
        loss = self.loss(tag_probs, targets)
        return loss

    def bk_forward(self, inputs, targets):
        tag_logits = self.bk_model(inputs)
        tag_probs = sanitize_probs(tag_logits)
        loss = self.loss(tag_probs, targets)
        return loss

    def predict(self, inputs, threshold=None):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            tag_logits = sanitize_probs(tag_logits)
        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            anomaly_id = self.label2id['Anomalous']
            pred_tags = np.zeros(probs.shape[0])
            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id

        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu()
        return pred_tags, tag_logits

    def collect_anomaly_scores(self, instances, vocab):
        anomaly_id = self.label2id['Anomalous']
        anomaly_scores = []
        gold_labels = []
        with torch.no_grad():
            self.model.eval()
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, vocab, False)
                tinst.to_cuda(device)
                tag_probs = sanitize_probs(self.model(tinst.inputs))
                anomaly_scores.extend(tag_probs[:, anomaly_id].detach().cpu().numpy().tolist())
                gold_labels.extend(self.label2id[inst.label] for inst in onebatch)
        return np.asarray(anomaly_scores, dtype=np.float64), np.asarray(gold_labels, dtype=np.int64)

    def _binary_metrics_from_scores(self, gold_labels, anomaly_scores, threshold):
        anomaly_id = self.label2id['Anomalous']
        gold_positive = gold_labels == anomaly_id
        pred_positive = anomaly_scores >= threshold

        TP = int(np.sum(pred_positive & gold_positive))
        TN = int(np.sum((~pred_positive) & (~gold_positive)))
        FP = int(np.sum(pred_positive & (~gold_positive)))
        FN = int(np.sum((~pred_positive) & gold_positive))

        precision = 100 * TP / (TP + FP) if TP + FP else 0
        recall = 100 * TP / (TP + FN) if TP + FN else 0
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0
        fpr = 100 * FP / (FP + TN) if FP + TN else 0

        return {
            'threshold': threshold,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'precision': precision,
            'recall': recall,
            'f': f,
            'fpr': fpr,
        }

    def _log_metrics(self, metrics):
        self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' %
                         (metrics['TP'], metrics['TN'], metrics['FN'], metrics['FP']))
        if metrics['TP'] + metrics['FP'] != 0:
            self.logger.info(
                'Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f, FPR = %.4f'
                % (
                    metrics['TP'],
                    metrics['TP'] + metrics['FP'],
                    metrics['precision'],
                    metrics['TP'],
                    metrics['TP'] + metrics['FN'],
                    metrics['recall'],
                    metrics['f'],
                    metrics['fpr'],
                )
            )
        else:
            self.logger.info('Precision is 0 and therefore f is 0')

    def tune_threshold(self, instances, vocab, threshold_min=0.1, threshold_max=0.9, threshold_step=0.01,
                       split_name='dev'):
        if threshold_step <= 0:
            raise ValueError('threshold_step must be positive.')
        if threshold_max < threshold_min:
            raise ValueError('threshold_max must be greater than or equal to threshold_min.')

        self.logger.info('Start threshold sweep on %s set: min=%.3f max=%.3f step=%.3f'
                         % (split_name, threshold_min, threshold_max, threshold_step))
        anomaly_scores, gold_labels = self.collect_anomaly_scores(instances, vocab)
        thresholds = np.arange(threshold_min, threshold_max + threshold_step * 0.5, threshold_step)

        best_metrics = None
        for threshold in thresholds:
            metrics = self._binary_metrics_from_scores(gold_labels, anomaly_scores, float(threshold))
            if best_metrics is None:
                best_metrics = metrics
                continue
            if metrics['f'] > best_metrics['f']:
                best_metrics = metrics
                continue
            if metrics['f'] == best_metrics['f'] and metrics['precision'] > best_metrics['precision']:
                best_metrics = metrics

        self.logger.info('Best threshold on %s set = %.3f' % (split_name, best_metrics['threshold']))
        self._log_metrics(best_metrics)
        return best_metrics['threshold'], best_metrics

    def evaluate(self, instances, threshold=0.5, vocab=None):
        if vocab is None:
            vocab = self.vocab
        self.logger.info('Start evaluating by threshold %.3f' % threshold)
        anomaly_scores, gold_labels = self.collect_anomaly_scores(instances, vocab)
        metrics = self._binary_metrics_from_scores(gold_labels, anomaly_scores, threshold)
        self._log_metrics(metrics)
        return metrics['precision'], metrics['recall'], metrics['f']


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', default='train', type=str, help='train or test')
    argparser.add_argument('--parser', default='IBM', type=str,
                           help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--min_cluster_size', type=int, default=100,
                           help="min_cluster_size.")
    argparser.add_argument('--min_samples', type=int, default=100,
                           help="min_samples")
    argparser.add_argument('--reduce_dimension', type=int, default=50,
                           help="Reduce dimentsion for fastICA, to accelerate the HDBSCAN probabilistic label estimation.")
    argparser.add_argument('--threshold', type=float, default=0.5,
                           help="Anomaly threshold.")
    argparser.add_argument('--beta', type=float, default=1.0,
                           help="weight for meta testing")
    argparser.add_argument('--backbone', type=str, default='gru',
                           help="Sequence encoder backbone: gru or bimamba.")
    argparser.add_argument('--mamba_state', type=int, default=mamba_state,
                           help="State expansion used by the BiMamba backbone.")
    argparser.add_argument('--mamba_conv', type=int, default=mamba_conv,
                           help="Convolution width used by the BiMamba backbone.")
    argparser.add_argument('--mamba_expand', type=int, default=mamba_expand,
                           help="Expansion factor used by the BiMamba backbone.")
    argparser.add_argument('--mamba_variant', type=str, default=mamba_variant,
                           help="Underlying Mamba block to use: auto, mamba, or mamba2.")
    argparser.add_argument('--auto_threshold', action='store_true',
                           help="Tune threshold on the BGL dev split before final evaluation.")
    argparser.add_argument('--threshold_min', type=float, default=0.1,
                           help="Lower bound for threshold sweep when auto_threshold is enabled.")
    argparser.add_argument('--threshold_max', type=float, default=0.9,
                           help="Upper bound for threshold sweep when auto_threshold is enabled.")
    argparser.add_argument('--threshold_step', type=float, default=0.01,
                           help="Step size for threshold sweep when auto_threshold is enabled.")

    args, extra_args = argparser.parse_known_args()

    parser = args.parser
    mode = args.mode
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    reduce_dimension = args.reduce_dimension
    threshold = args.threshold
    beta = args.beta
    backbone = args.backbone
    mamba_state = args.mamba_state
    mamba_conv = args.mamba_conv
    mamba_expand = args.mamba_expand
    mamba_variant = args.mamba_variant
    auto_threshold = args.auto_threshold
    threshold_min = args.threshold_min
    threshold_max = args.threshold_max
    threshold_step = args.threshold_step

    # process BGL
    dataset = 'BGL'
    # Mark results saving directories.
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    prob_label_res_file_BGL = os.path.join(save_dir,
                                       'results/MetaLog/' + dataset + '_' + parser +
                                       '/prob_label_res/mcs-' + str(min_cluster_size) + '_ms-' + str(min_samples))
    rand_state_BGL = os.path.join(save_dir,
                              'results/MetaLog/' + dataset + '_' + parser +
                              '/prob_label_res/random_state')

    output_model_dir = os.path.join(save_dir, 'models/MetaLog/' + dataset + '_' + parser + '/model')
    output_res_dir = os.path.join(save_dir, 'results/MetaLog/' + dataset + '_' + parser + '/detect_res')

    # Training, Validating and Testing instances.
    template_encoder_BGL = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
    processor_BGL = Preprocessor()
    train_BGL, dev_BGL, test_BGL = processor_BGL.process(dataset=dataset, parsing=parser, cut_func=cut_by_316_filter,
                                                         template_encoding=template_encoder_BGL.present)

    # Log sequence representation.
    sequential_encoder_BGL = Sequential_TF(processor_BGL.embedding)
    train_reprs_BGL = sequential_encoder_BGL.present(train_BGL)
    for index, inst in enumerate(train_BGL):
        inst.repr = train_reprs_BGL[index]
    test_reprs_BGL = sequential_encoder_BGL.present(test_BGL)
    for index, inst in enumerate(test_BGL):
        inst.repr = test_reprs_BGL[index]

    # Dimension reduction if specified.
    transformer_BGL = None
    if reduce_dimension != -1:
        start_time = time.time()
        print("Start FastICA, target dimension: %d" % reduce_dimension)
        transformer_BGL = FastICA(n_components=reduce_dimension)
        train_reprs_BGL = transformer_BGL.fit_transform(train_reprs_BGL)
        for idx, inst in enumerate(train_BGL):
            inst.repr = train_reprs_BGL[idx]
        print('Finished at %.2f' % (time.time() - start_time))

    # Probabilistic labeling.
    # Sample normal instances.
    train_normal_BGL = [x for x, inst in enumerate(train_BGL) if inst.label == 'Normal']
    normal_ids_BGL = train_normal_BGL[:int(0.5 * len(train_normal_BGL))]
    label_generator_BGL = Probabilistic_Labeling(min_samples=min_samples, min_clust_size=min_cluster_size,
                                             res_file=prob_label_res_file_BGL, rand_state_file=rand_state_BGL)
    labeled_train_BGL = label_generator_BGL.auto_label(train_BGL, normal_ids_BGL)

    # Below is used to test if the loaded result match the original clustering result.
    TP, TN, FP, FN = 0, 0, 0, 0

    for inst in labeled_train_BGL:
        if inst.predicted == 'Normal':
            if inst.label == 'Normal':
                TN += 1
            else:
                FN += 1
        else:
            if inst.label == 'Anomalous':
                TP += 1
            else:
                FP += 1
    from utils.common import get_precision_recall

    print(len(normal_ids_BGL))
    print('TP %d TN %d FP %d FN %d' % (TP, TN, FP, FN))
    p, r, f = get_precision_recall(TP, TN, FP, FN)
    print('%.4f, %.4f, %.4f' % (p, r, f))

    # Load Embeddings
    vocab_BGL = Vocab()
    vocab_BGL.load_from_dict(processor_BGL.embedding)

    # process HDFS
    dataset = 'HDFS'
    # Mark results saving directories.
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    prob_label_res_file_HDFS = os.path.join(save_dir,
                                       'results/MetaLog/' + dataset + '_' + parser +
                                       '/prob_label_res/mcs-' + str(min_cluster_size) + '_ms-' + str(min_samples))
    rand_state_HDFS = os.path.join(save_dir,
                              'results/MetaLog/' + dataset + '_' + parser +
                              '/prob_label_res/random_state')

    # Training, Validating and Testing instances.
    template_encoder_HDFS = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
    processor_HDFS = Preprocessor()
    train_HDFS, _, _ = processor_HDFS.process(dataset=dataset, parsing=parser, cut_func=cut_by_415,
                                         template_encoding=template_encoder_HDFS.present)

    # Log sequence representation.
    sequential_encoder_HDFS = Sequential_TF(processor_HDFS.embedding)
    train_reprs_HDFS = sequential_encoder_HDFS.present(train_HDFS)
    for index, inst in enumerate(train_HDFS):
        inst.repr = train_reprs_HDFS[index]

    # Dimension reduction if specified.
    transformer_HDFS = None
    if reduce_dimension != -1:
        start_time = time.time()
        print("Start FastICA, target dimension: %d" % reduce_dimension)
        transformer_HDFS = FastICA(n_components=reduce_dimension)
        train_reprs_HDFS = transformer_HDFS.fit_transform(train_reprs_HDFS)
        for idx, inst in enumerate(train_HDFS):
            inst.repr = train_reprs_HDFS[idx]
        print('Finished at %.2f' % (time.time() - start_time))

    labeled_train_HDFS = train_HDFS

    # aggregate vocab and label2id
    vocab = Vocab()
    new_embedding = {}
    for key in processor_BGL.embedding.keys():
        new_embedding[key] = processor_BGL.embedding[key]
    for key in processor_HDFS.embedding.keys():
        new_embedding[key + 432] = processor_HDFS.embedding[key]
    # Load Embeddings
    vocab_HDFS = Vocab()
    vocab_HDFS.load_from_dict(processor_HDFS.embedding)
    print(new_embedding.keys())
    vocab.load_from_dict(new_embedding)

    metalog = MetaLog(
        vocab,
        num_layer,
        lstm_hiddens,
        processor_BGL.label2id,
        backbone=backbone,
        mamba_state=mamba_state,
        mamba_conv=mamba_conv,
        mamba_expand=mamba_expand,
        mamba_variant=mamba_variant,
    )

    # meta learning
    log = 'backbone={}_layer={}_hidden={}_epoch={}'.format(backbone, num_layer, lstm_hiddens, epochs)
    best_model_file = os.path.join(output_model_dir, log + '_best.pt')
    last_model_file = os.path.join(output_model_dir, log + '_last.pt')
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    if mode == 'train':
        # Train
        model_lr = get_backbone_lr(backbone)
        optimizer = Optimizer(filter(lambda p: p.requires_grad, metalog.model.parameters()), lr=model_lr)
        global_step = 0
        bestF = 0
        for epoch in range(epochs):
            metalog.model.train()
            metalog.bk_model.train()
            start = time.strftime("%H:%M:%S")
            metalog.logger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %s" %
                               (epoch, start, optimizer.lr))

            batch_num = int(np.ceil(len(labeled_train_HDFS) / float(batch_size)))
            batch_iter = 0
            batch_num_test = int(np.ceil(len(labeled_train_BGL) / float(batch_size)))
            batch_iter_test = 0
            total_bn = max(batch_num, batch_num_test)
            meta_train_loader = data_iter(labeled_train_HDFS, batch_size, True)
            meta_test_loader = data_iter(labeled_train_BGL, batch_size, True)

            for i in range(total_bn):
                optimizer.zero_grad()
                # meta train
                meta_train_batch = meta_train_loader.__next__()
                meta_test_batch = meta_test_loader.__next__()
                tinst_tr = generate_tinsts_binary_label(meta_train_batch, vocab_HDFS)
                tinst_tr.to_cuda(device)
                loss = metalog.forward(tinst_tr.inputs, tinst_tr.targets)
                loss_value = loss.data.cpu().numpy()
                loss.backward(retain_graph=True)
                batch_iter += 1
                metalog.bk_model = get_updated_network(metalog.model, metalog.bk_model, model_lr).train().cuda()
                # meta test
                tinst_test = generate_tinsts_binary_label(meta_test_batch, vocab_BGL)
                tinst_test.to_cuda(device)
                loss_te = beta * metalog.bk_forward(tinst_test.inputs, tinst_test.targets)
                loss_value_te = loss_te.data.cpu().numpy() / beta
                loss_te.backward()
                batch_iter_test += 1
                # aggregate
                optimizer.step()
                global_step += 1
                if global_step % 500 == 0:
                    metalog.logger.info("Step:%d, Epoch:%d, meta train loss:%.2f, meta test loss:%.2f" \
                                       % (global_step, epoch, loss_value, loss_value_te))
                if batch_iter == batch_num:
                    meta_train_loader = data_iter(labeled_train_HDFS, batch_size, True)
                    batch_iter = 0
                if batch_iter_test == batch_num_test:
                    meta_test_loader = data_iter(labeled_train_BGL, batch_size, True)
                    batch_iter_test = 0
               
            if test_BGL:
                eval_threshold = threshold
                selection_f = None
                if auto_threshold and dev_BGL:
                    eval_threshold, dev_metrics = metalog.tune_threshold(
                        dev_BGL,
                        vocab_BGL,
                        threshold_min=threshold_min,
                        threshold_max=threshold_max,
                        threshold_step=threshold_step,
                        split_name='dev',
                    )
                    selection_f = dev_metrics['f']
                    metalog.logger.info('Testing on test set with tuned threshold %.3f selected from dev.'
                                        % eval_threshold)
                else:
                    metalog.logger.info('Testing on test set.')
                _, _, test_f = metalog.evaluate(test_BGL, eval_threshold, vocab=vocab_BGL)
                if selection_f is None:
                    selection_f = test_f
                if selection_f > bestF:
                    metalog.logger.info("Exceed best selection f: history = %.2f, current = %.2f"
                                        % (bestF, selection_f))
                    torch.save(metalog.model.state_dict(), best_model_file)
                    bestF = selection_f
            metalog.logger.info('Training epoch %d finished.' % epoch)
            torch.save(metalog.model.state_dict(), last_model_file)

    def evaluate_checkpoint(model_file, title):
        if not os.path.exists(model_file):
            return
        metalog.logger.info('=== %s ===' % title)
        metalog.model.load_state_dict(torch.load(model_file))
        eval_threshold = threshold
        if auto_threshold and dev_BGL:
            eval_threshold, _ = metalog.tune_threshold(
                dev_BGL,
                vocab_BGL,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                split_name='dev',
            )
            metalog.logger.info('Use tuned threshold %.3f for %s evaluation.' % (eval_threshold, title))
        metalog.evaluate(test_BGL, eval_threshold, vocab=vocab_BGL)

    evaluate_checkpoint(last_model_file, 'Final Model')
    evaluate_checkpoint(best_model_file, 'Best Model')
    metalog.logger.info('All Finished')
