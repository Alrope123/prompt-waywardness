import os
import argparse
import pickle as pkl
import random
import torch
import math
import logging
import numpy as np

from collections import Counter, defaultdict

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from data import load_data, prepare_data, load_prompt, output_metrices
from run import train, inference
from model_util import load_checkpoint, set_extra_embeddings, \
    set_separate_lm_head, set_transformed_lm_head
from util import get_prompts, get_paths, flatten_label_losses, \
    prepend_task_tokens, reassign_output_tokens, f1_score


N_LABELS_DICT = {"SST-2": 2, "sst-5": 5, "mr": 2, "cr": 2, "mpqa": 2,
                 "subj": 2, "trec": 6, "trec-5": 5, "trec-4": 4, "trec-3": 3, "CoLA": 2,
                 "amazon": 5, "yelp_full": 5, "yelp_binary": 2,
                 "agnews": 4, "copa": 2, "boolq": 2,
                 "RTE": 2, "cb": 3,
                 "yahoo": 10, "dbpedia": 14, 'climate_fever': 4, 
                 'ethos-national_origin': 2, 'ethos-race': 2,
                 'ethos-religion': 2, 'financial_phrasebank': 3, 
                 'hate_speech18': 2, 'medical_questions_pairs': 2, 
                 'poem_sentiment': 4, 'superglue-cb': 3, 
                 'tweet_eval-hate': 2, 'tweet_eval-stance_atheism': 3, 
                 'tweet_eval-stance_feminist': 3, 'anli': 3, 
                 'glue-mnli': 3, 'glue-qnli': 2, 'glue-rte': 2, 
                 'glue-wnli': 2, 'scitail': 2, 'sick': 3,
                 'ai2_arc': 4, 'codah': 4, 'commonsense_qa': 5, 
                 'openbookqa': 4, 'qasc': 8, 'quarel': 2, 'quartz-no_knowledge': 2, 
                 'quartz-with_knowledge': 2, 'superglue-copa': 2, 'wino_grande': 2
}

PROMPT_DICT = {
    "NI": ['subtask047_misc_answering_science_questions', 'subtask034_winogrande_question_modification_object',
            'subtask028_drop_answer_generation', 'subtask054_multirc_write_correct_answer', 
            'subtask019_mctaco_temporal_reasoning_category', 'subtask021_mctaco_grammatical_logical', 
            'subtask027_drop_answer_type_generation', 'subtask038_qasc_combined_fact', 
            'subtask029_winogrande_full_object', 'subtask033_winogrande_answer_generation', 
            'subtask044_essential_terms_identifying_essential_words', 'subtask050_multirc_answerability', 
            'subtask061_ropes_answer_generation', 'subtask002_quoref_answer_generation', 
            'subtask037_qasc_generate_related_fact', 'subtask046_miscellaenous_question_typing', 
            'subtask057_multirc_classify_incorrect_answer', 'subtask058_multirc_question_answering', 
            'subtask006_mctaco_question_generation_transient_stationary', 
            'subtask020_mctaco_span_based_question', 'subtask040_qasc_question_generation', 
            'subtask042_qasc_incorrect_option_generation', 'subtask008_mctaco_wrong_answer_generation_transient_stationary', 
            'subtask023_cosmosqa_question_generation', 'subtask025_cosmosqa_incorrect_answer_generation', 
            'subtask039_qasc_find_overlapping_words', 'subtask045_miscellaneous_sentence_paraphrasing', 
            'subtask060_ropes_question_generation', 'subtask007_mctaco_answer_generation_transient_stationary', 
            'subtask013_mctaco_answer_generation_absolute_timepoint', 'subtask059_ropes_story_generation', 
            'subtask048_multirc_question_generation'],
    "PILE": ['prompt00', 'prompt01', 'prompt02', 'prompt03', 'prompt04', 'prompt05', 'prompt06', 'prompt07', 
            'prompt08', 'prompt09', 'prompt10', 'prompt11', 'prompt12', 'prompt13', 'prompt14', 'prompt15', 
            'prompt16', 'prompt17', 'prompt18', 'prompt19', 'prompt20', 'prompt21', 'prompt22', 'prompt23', 
            'prompt24', 'prompt25', 'prompt26', 'prompt27', 'prompt28', 'prompt29'],
    "TRUE": {
        "SST-2": ["SST-2_0", "SST-2_1", "SST-2_2", "SST-2_3", "SST-2_4"],
        "sst-5": ["sst-5_0", "sst-5_1", "sst-5_2", "sst-5_3", "sst-5_4"],
        "agnews": ["agnews_0", "agnews_1", "agnews_2", "agnews_3", "agnews_4"],
        "trec": ["trec_0", "trec_1", "trec_2", "trec_3", "trec_4"],
        "subj": ["subj_0", "subj_1", "subj_2", "subj_3", "subj_4"]
    },
    "TEST": ['prompt00', 'prompt01', 'prompt02']
}


def main(logger, args):
    args.gpt2 = args.gpt2.replace("gpt2-small", "gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
    model = None

    if args.train_task is None:
        # standard case where the training task and the test task are the same
        train_task = args.task
    else:
        # zero-shot transfer case where the training task is different from the test task
        train_task = args.train_task
        assert args.do_check

    # datasets where the average input length is long
    long_datasets = ["cr", "subj", "agnews",
                     "amazon", "yelp_full", "yelp_binary", "boolq",
                     "dbpedia", "yahoo", 'climate_fever', 
                     'ethos-national_origin', 'ethos-race', 
                     'ethos-religion', 'financial_phrasebank', 'hate_speech18', 
                     'medical_questions_pairs', 'superglue-cb', 
                     'tweet_eval-hate', 'anli', 'glue-mnli', 'glue-qnli', 
                     'glue-rte', 'glue-wnli', 'scitail', "ai2_arc", "codah", "openbookqa",
                     'quarel', 'quartz-with_knowledge']
    max_length = 256 if train_task in long_datasets else 128
    batch_size = int(args.batch_size / 2) if train_task in long_datasets else args.batch_size

    logger.info("%s %s" % (args.method, args.task))

    assert args.method in ["direct", "channel"]

    if args.use_demonstrations:
        assert args.do_zeroshot and not args.do_train

    if args.ensemble:
        assert args.use_demonstrations

    k = int(args.k)
    seed = int(args.seed)

    train_data = load_data(args.data_dir, train_task, k, seed, "train")
    if args.split is None:
        assert args.do_zeroshot
        dev_data = None
    else:
        dev_data = load_data(args.data_dir, args.task, k, seed, args.split)
    
    gammas = [float(gamma) for gamma in args.gamma.split(",")]
    tseeds = [int(tseed) for tseed in args.train_seed.split(",")]
    template_idx = 1

    if args.prompt_group is not None: 
        assert args.prompt_tune and args.init_method == "manual"
        if args.prompt_group == "TRUE":
            prompt_names = PROMPT_DICT[args.prompt_group][args.task]
        else:
            prompt_names = PROMPT_DICT[args.prompt_group]
        prompts = [load_prompt(args.prompts_dir, prompt_name, int(args.pile_len)) for prompt_name in prompt_names]
    elif args.prompt_task is not None:
        assert args.prompt_tune and args.init_method == "manual"
        prompts = [load_prompt(args.prompts_dir, args.prompt_task, int(args.pile_len))]
    else:
        prompts = [None]

    if args.prompt_group is not None: 
        total_results = {"accuracy": [], "prompt-f1": []}
    for prompt in prompts:
        gamma_results = []
        for gamma in gammas:
            seed_results = []
            for tseed in tseeds:
                acc, f1, mapped_prompt, norm_distance, mapped_acc, mapped_f1 = run(logger, args.do_train, args.do_zeroshot,
                                args.task, train_task, args.prompt_task,
                                k, seed, tseed,
                                args.out_dir, args.split,
                                tokenizer, model, train_data, dev_data,
                                batch_size, max_length, args.gpt2, args.init_method, args.prefix_type,
                                template_idx, args.method,
                                args.lr, gamma,
                                args.warmup_steps, args.num_training_steps, args.eval_period,
                                prompt,
                                use_demonstrations=args.use_demonstrations,
                                use_calibration=args.use_calibration,
                                ensemble=args.ensemble,
                                is_null=args.split is None,
                                prompt_tune=args.prompt_tune,
                                head_tune=args.head_tune,
                                transform_tune=args.transform_tune,
                                bad=args.bad,
                                do_check=args.do_check,
                                n_prefix=args.n_prefix,
                                f1_threshold=args.f1_threshold,
                                pile_len=args.pile_len)
                prompt_f1 = f1_score(mapped_prompt, prompt)
                seed_results.append({
                    "soft_accuracy": acc,
                    "prompt_f1": prompt_f1
                })
            gamma_results.append({
                "acc" : np.average([seed_result["soft_accuracy"] for seed_result in seed_results]),
                "prompt_f1": np.average([seed_result["prompt_f1"] for seed_result in seed_results])
            })
        
        # select the best one over the f1 threshold
        sorted_gamma_results = sorted(gamma_results, key=lambda x:x["acc"], reverse=True)
        best_idx = 0
        for i, result in enumerate(sorted_gamma_results):
            if result["prompt_f1"] > args.f1_threshold:
                best_idx = i
                break
        best_results = sorted_gamma_results[best_idx]
        total_results["accuracy"].append(best_results["acc"])
        total_results["prompt-f1"].append(best_results["prompt_f1"])

        logger.info("Results for prompt: {}".format(prompt))
        logger.info("Accuracy = %.1f" % (100 * best_results["acc"]))
        logger.info("Prompt_F1 = %.2f" % (best_results["prompt_f1"]))
    
    if args.prompt_group is not None:
        logger.info("Results for task {} for {} prompts:".format(args.task, args.prompt_group))
        logger.info("Accuracy = %.1f" % (100 * np.average([total_results["accuracy"]])))
        logger.info("Prompt_F1 = %.2f" % (np.average(total_results["prompt-f1"])))
        

def run(logger, do_train, do_zeroshot, task, train_task, prompt_task,
        k, seed, train_seed,
        out_dir, split, tokenizer, model,
        train_data, dev_data,
        batch_size, max_length, gpt2, init_method, prefix_type,
        template_idx, method_type, learning_rate, 
        gamma,
        warmup_steps, num_training_steps, eval_period,
        prompt,
        use_demonstrations=False,
        use_calibration=False,
        ensemble=False,
        is_null=False,
        prompt_tune=False,
        head_tune=False,
        transform_tune=False,
        bad=False,
        do_check=False, n_prefix=-1,
        f1_threshold=0.95, pile_len=-1):

    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)

    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(train_seed)

    if head_tune or transform_tune:
        assert method_type == "direct"

    if init_method == "manual":
        n_prefix = len(tokenizer(prompt)["input_ids"]) if n_prefix < 0 else n_prefix
    elif init_method == "vocab":
        n_prefix = 20

    n_classes = N_LABELS_DICT.get(task, None)
    templates = get_prompts(task, template_idx)

    n_classes_train = N_LABELS_DICT.get(train_task, None)
    templates_train = get_prompts(train_task, template_idx)

    if task in ["yelp_full", "amazon"] and train_task in ["SST-2", "mr", "cr"]:
        templates = [t.replace(".", " .") for t in templates]

    max_length_per_example = max_length

    if use_demonstrations and not ensemble:
        assert do_zeroshot and not do_train
        mem = batch_size * max_length
        if n_classes == 2:
            max_length = max_length * k
        elif n_classes in [4, 5]:
            max_length = int(max_length * 1.5 * k)
        elif n_classes in [6]:
            max_length = int(max_length * 2 * k)
        else:
            max_length = 1024

        max_length = min(max_length, 1024)
        batch_size = int(mem / max_length)


    if do_zeroshot:
        cache_paths = [get_paths(out_dir, gpt2, method_type, task, do_zeroshot,
                                 k, seed, train_seed, split, template_idx,
                                 use_demonstrations=use_demonstrations,
                                 ensemble=ensemble)]
        checkpoints = [None]

    else:
        out_dir = get_paths(out_dir, gpt2, method_type, train_task, do_zeroshot,
                            k, seed, train_seed, split, template_idx,
                            batch_size, learning_rate, warmup_steps,
                            gamma,
                            init_method, prompt_task,
                            use_demonstrations=use_demonstrations,
                            ensemble=ensemble,
                            bad=bad,
                            prompt_tune=prompt_tune,
                            head_tune=head_tune,
                            transform_tune=transform_tune,
                            n_prefix=n_prefix,
                            f1_threshold=f1_threshold,
                            prompt_file_len=pile_len)

        k = int(k)

        cache_paths = [os.path.join(out_dir, "{}cache-{}-{}-{}.pkl".format(
            task + "-" if train_task != task else "",
            split, num_training_steps, prefix_type))]
        checkpoints = [os.path.join(out_dir, "model-{}.pt".format(num_training_steps))]

    mapping = None

    if do_train and (head_tune or not do_check):

        inputs = prepare_data(
            tokenizer, None, train_data,
            max_length=max_length,
            max_length_per_example=max_length_per_example,
            n_classes=n_classes_train,
            templates=templates_train,
            method_type=method_type,
            is_training=True,
            ensemble=ensemble)


        logger.info(out_dir)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)


        if not do_check:	

            model = GPT2LMHeadModel.from_pretrained(gpt2)	

            if prompt_tune:	
                for param in model.parameters():
                        param.requires_grad = False

                if init_method == "manual":
                    prompt_ids = tokenizer(prompt)["input_ids"]
                    set_extra_embeddings(model, n_prefix, prompt_ids)
                    logger.info("Using a prompt of size {}".format(len(prompt_ids)))
                else:
                    set_extra_embeddings(model, n_prefix, init_method)
                inputs = prepend_task_tokens(tokenizer, inputs, n_prefix)

            elif head_tune:	
                mapping, inputs = reassign_output_tokens(inputs, for_labels=True)	
                logger.info("Created mapping with {} vocabs".format(len(mapping)))	
                set_separate_lm_head(model, mapping)	
                for param in model.parameters():	
                    param.requires_grad = False	
                for param in model.lm_head.my_lm_head.parameters():	
                    param.requires_grad = True	

            elif transform_tune:	
                set_transformed_lm_head(model)	
                for param in model.parameters():	
                    param.requires_grad = False	
                for param in model.lm_head.transform.parameters():	
                    param.requires_grad = True	

            model = model.cuda()	

            if torch.cuda.device_count() > 1:	
                model = torch.nn.DataParallel(model)	

            train(logger, model, inputs, batch_size, out_dir,	
                  learning_rate=learning_rate,	
                  warmup_steps=warmup_steps,	
                  eval_period=eval_period,	
                  num_training_steps=num_training_steps,	
                  prompt_tune=prompt_tune,	
                  head_tune=head_tune,	
                  transform_tune=transform_tune)

    input_tensors = prepare_data(
        tokenizer, train_data, dev_data,
        max_length=max_length,
        max_length_per_example=max_length_per_example,
        n_classes=n_classes,
        templates=templates,
        method_type=method_type,
        use_demonstrations=use_demonstrations,
        ensemble=ensemble,
        is_null=is_null)


    if prompt_tune:
        input_tensors = prepend_task_tokens(tokenizer, input_tensors, n_prefix)

    if head_tune:
        # some tricks in case train_task and test_task are different
        if task != train_task:
            if task in ["sst-5", "yelp_full", "amazon"] and train_task in ["SST-2", "mr", "cr"]:
                input_tensors = [input_tensors[0], input_tensors[-1]]
                if head_tune:
                    label_counter = {'0': '0', '4': '1'}
                    dev_data = [(x, label_counter.get(y, '-1')) for x, y in dev_data]
            elif task in ["SST-2", "mr"] and train_task in ["SST-2", "mr", "sst-5"]:
                pass
            else:
                raise NotImplementedError()

        if mapping is None:
            mapping, inputs = reassign_output_tokens(inputs, for_labels=head_tune)

        train_labels = set([label for _, label in train_data])
        if len(train_labels) != n_classes:
            train_labels = sorted(train_labels)
            input_tensors = [input_tensors[int(l)] for l in train_labels]
            dev_data = [(sent, str(train_labels.index(l)) if l in train_labels else -1)
                        for sent, l in dev_data]

        _, input_tensors = reassign_output_tokens(input_tensors, for_labels=head_tune,
                                                  mapping={v: k for k, v in mapping.items()})
        logger.info(mapping)
        logger.info("Checked that train mapping and test mapping are identical")


    # for debugging ...
    logger.info("Checking the first example...")
    input_ids = input_tensors[0]["input_ids"][0].numpy().tolist()
    token_type_ids = input_tensors[0]["token_type_ids"][0].numpy().tolist()
    logger.info("Input:")
    logger.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
    logger.info("Output:")
    logger.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))

    results = []
    for cache_path, checkpoint in zip(cache_paths, checkpoints):

        logger.info(cache_path)

        # if there is a cache, load it
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                losses = pkl.load(f)
        else:
            if checkpoint is not None and not os.path.exists(checkpoint):
                logger.info("checkpoint %s not found..." % checkpoint)
                assert False

            if checkpoint is None and model is not None and do_zeroshot:
                logger.info("Reusing the loaded model...")
                pass
            else:
                logger.info("Loading the model")
                torch.cuda.empty_cache()
                del model
                model = load_checkpoint(gpt2, checkpoint,
                                        prompt_tune=prompt_tune,
                                        head_tune=head_tune,
                                        transform_tune=transform_tune,
                                        n_prefix=n_prefix,
                                        mapping=mapping)
                
                if prefix_type == "discrete":
                    prefix_ids, aux_loss = model.transformer.wte.map_to_discrete()
                    logger.info("The mapped discrete prefix is: {}".format(tokenizer.decode(prefix_ids)))
                    logger.info("The norm distance is: {}".format(aux_loss))

                model = model.cuda()
                model.eval()
                logger.info("Finished loading the model")

            losses = []
            for input_tensor in input_tensors:
                losses.append(inference(model,
                                        input_tensor,
                                        batch_size * 8,
                                        bad=bad))

            with open(cache_path, "wb") as f:
                pkl.dump(losses, f)

        if is_null:
            continue

        if ensemble:
            losses = flatten_label_losses(losses, dev_data)

        if use_calibration:
            bias_path = cache_path.replace(split, "None")
            assert os.path.exists(bias_path), bias_path
            with open(bias_path, "rb") as f:
                bias_losses = pkl.load(f)

            for i, (bias_loss, loss) in enumerate(zip(bias_losses, losses)):
                loss = np.array(loss)
                bias_loss = np.array(bias_loss)
                if ensemble:
                    bias_loss = bias_loss.reshape(1, -1)
                losses[i] = loss - bias_loss

       
        acc, f1 = evaluate(dev_data, {str(i): loss for i, loss in enumerate(losses)})
        logger.info(acc)
        logger.info(f1)

        if hasattr(model, 'module'):
            prefix_ids, aux_loss = model.module.transformer.wte.map_to_discrete()
        else:
            prefix_ids, aux_loss = model.transformer.wte.map_to_discrete()
        logger.info("The mapped discrete prefix is: {}".format(tokenizer.decode(prefix_ids)))
        logger.info("The norm distance is: {}".format(aux_loss))

        logger.info("Evaluating mapped discrete prompt")
        mapped_losses = []
        for i, input_tensor in enumerate(input_tensors):
            mapped_losses.append(inference(model,
                                    input_tensor,
                                    batch_size * 8,
                                    bad=bad))
        mapped_acc, mapped_f1 = evaluate(dev_data, {str(i): loss for i, loss in enumerate(mapped_losses)})
        logger.info(mapped_acc)
        logger.info(mapped_f1)

        return acc, f1, tokenizer.decode(prefix_ids), aux_loss.item(), mapped_acc, mapped_f1


def evaluate(dev_data, label_losses, is_classification=True):
    if type(label_losses)==list:
        label_losses = np.array(label_losses)
    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for idx, (_, label) in enumerate(dev_data):
        label_loss = {l:np.sum(label_losses[l][idx]) for l in label_losses}
        prediction = sorted(label_loss.items(), key=lambda x: x[1])[0][0]
        accs.append(prediction==label)
        precisions[prediction].append(prediction==label)
        recalls[label].append(prediction==label)

    if not is_classification:
        return np.mean(accs)

    f1s = []
    for key in recalls:
        precision = np.mean(precisions[key]) if key in precisions else 1.0
        recall = np.mean(recalls[key])
        if precision+recall==0:
            f1s.append(0)
        else:
            f1s.append(2*precision*recall / (precision+recall))

    return np.mean(accs), np.mean(f1s)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_zeroshot", default=False, action="store_true")
    parser.add_argument("--do_check", default=False, action="store_true")

    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--ensemble", default=False, action="store_true")
    parser.add_argument("--prompt_tune", default=False, action="store_true")
    parser.add_argument("--head_tune", default=False, action="store_true")
    parser.add_argument("--transform_tune", default=False, action="store_true")
    parser.add_argument("--bad", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default="SST-2")
    parser.add_argument("--train_task", type=str, default=None)
    parser.add_argument("--prompt_task", type=str, default=None)
    parser.add_argument("--prompt_group", type=str, default=None)

    parser.add_argument("--k", type=str, default="-1")
    parser.add_argument("--seed", type=str, default="100")
    parser.add_argument("--train_seed", type=str, default="1,10,100")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gamma", type=str, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_training_steps", type=int, default=2000)
    parser.add_argument("--n_templates", type=int, default=1)
    parser.add_argument("--eval_period", type=int, default=500)
    parser.add_argument("--f1_threshold", type=float, default=0.98)
    parser.add_argument("--pile_len", type=int, default=-1)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--prompts_dir", type=str, default="prompts")

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--n_prefix", type=int, default=-1)
    parser.add_argument("--gpt2", type=str, default="gpt2-large")
    parser.add_argument("--init_method", type=str, default="manual")
    parser.add_argument("--prefix_type", type=str, default="soft")

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
