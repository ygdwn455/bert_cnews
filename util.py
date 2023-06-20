from sklearn.metrics import f1_score
from transformers import DataProcessor, InputExample, InputFeatures
import os
import torch
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import logging
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
import numpy as np
import jieba

logger = logging.getLogger(__name__)

Contradictory_keys = {'经济':['经济','经济增长','总需求','GDP','经济运行','内需','外需','可持续性','经济指标','总体经济','增长','内生',
                            '居民收入','宏观政策','经济体','动能','内生性','马车','财政政策','韧性','经济趋势'],
                    '流动':['流动性','资金面','资金紧张','资金','货币','钱荒','利率','对冲','公开市场','时点','降准','货币政策','压力',
                           'MLF','市场','准备金','中性','正回购','储率','超储率','资金量'],
                    '通胀':['通胀','物价','CPI','物价水平','通胀率','价格水平','货币政策','PCE','PPI','总需求','降息',
                          '商品价格','预期','斜率','锚定','薪资','失业率','能源价格','QE','触顶','名义工资']}
Contradictory_inds = ['经济','流动','通胀']


def simple_accuracy(preds, labels):
    return (preds == labels).mean()



def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    labels = labels.astype(float)
    preds = preds.astype(float)
    f1 = f1_score(y_true=labels, y_pred=preds,average='weighted')
    return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
    }


def Identify_contradiction(text):
    text = str(text)
    seg_list = jieba.cut(text)

    list01_num = 0
    list02_num = 0
    list03_num = 0
    for word in seg_list:
        if word in Contradictory_keys['经济']:
            list01_num += 1
        elif word in Contradictory_keys['流动']:
            list02_num += 1
        elif word in Contradictory_keys['通胀']:
            list03_num += 1
    tmp = np.array([list01_num,list02_num,list03_num])
    return Contradictory_inds[tmp.argmax()]

class CnesProcessor(DataProcessor):
    """Processor for the cnews data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "cnews.train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "cnews.test.txt")), "dev")

    def get_labels(self):
        """See base class."""
        return ['经济正','经济负','流动正','流动负','通胀正','通胀负']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        'bert',
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                #pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                #pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                #pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        # if args.local_rank in [-1, 0]:
        #     logger.info("Saving features into cached file %s", cached_features_file)
        #     torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


processors["cnews"] = CnesProcessor
output_modes["cnews"] = "classification"

