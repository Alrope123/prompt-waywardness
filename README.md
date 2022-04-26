# Prompt Waywardness

This includes an original implementation "[PROMPT WAYWARDNESS: The Curious Case of Discretized Interpretation of Continuous Prompts][paper]" by [Daniel Khashabi][danielk], [Xinxi Lyu][xinxil], [Sewon Min][sewonm], [Lianhui Qin][lianhuiq], [Kyle Richardson][kyler], [Sameer Singh][sameers], [Sean Welleck][seanw], [Hannaneh Hajishirzi][hannanehh], [Tushar Khot][tushark], [Ashish Sabharwal][ashishs], [Yejin Choi][yejinc].

This code provides commands to run the models and reproduce the numbers reported in the paper, based on the [Channel LM Prompting][channel-lm-code] codebase.

Please leave issues for any questions about the paper or the code.

If you find our code or paper useful, please cite the paper:
```bibtex 
@inproceedings{khashabi2021waywardness,
  title={{PROMPT WAYWARDNESS: The Curious Case of Discretized Interpretation of Continuous Prompts}},
  author = {Khashabi, Daniel and Lyu, Shane and Min, Sewon and Qin, Lianhui and Richardson, Kyle and Singh, Sameer and Welleck, Sean and Hajishirzi, Hannaneh and Khot, Tushar and Sabharwal, Ashish and Choi, Yejin},
  booktitle={Proceedings of NAACL},
  year={2022}
}
```

## Content

1. [Installation](#installation)
2. [Download & Preprocess Data](#download-and-preprocess-data)
3. [Default Commands](#default-commands)
4. [Reproducing Main Results](#reproducing-main-results) (Section 4.2 of the paper)
    * [Prompt tuning constrained by projections](#prompt-tuning-constrained-by-projections)
    * [Unconstrained prompt tuning](#unconstrained-prompt-tuning)
5. [Reproducing Analysis](#reproducing-analysis) (Section 4.3 of the paper)
    * [Effect of Gamma](#effect-of-gamma)
    * [Effect of Prompt Length](#effect-of-prompt-length)
    * [Effect of Model Size](#effect-of-model-size)
    * [Projection onto true task definitions](#projection-onto-true-task-definitions)

You can run the channel model and the direct model for each of these methods. Please see Section 3 of the [paper][paper] for more details about these formulations.

## Installation

```
$ conda create -n waywardness python=3.8
$ conda activate waywardness
$ conda install pytorch=1.7.1 -c pytorch
$ pip install transformers==4.3.0
```

## Download and Preprocess Data

We use (and modify) the data and the preprocessing script from Gao et al. ACL 2021 ([paper][lm-bff-paper], [code][lm-bff-code]) and Zhang et al. NeurIPS 2015 ([paper][zhang-paper], [data][zhang-data]).

**To download the k-shot data (already preprocessed):**
Download the data (65.6MB) from [this link](https://drive.google.com/file/d/1a2Y2SdwfTvX_obsY5AiLrS-mQGcbZrfo/view?usp=sharing). Pleae place `data-processed.zip` under the same directory as the code and unzip it.

**To download the original data and preprocess yourself:**
Download the data (14MB) from [this link](https://drive.google.com/file/d/1y_BV9qAiRz72JrRO1jlK6IQhvuG1k_YG/view?usp=sharing). Pleae place `data-processed.zip` under the same directory as the code and unzip it.

Then, run `python3 generative_k_shot_data.py`, and you are done!

Optionally, you can specify arguments such as
* `--data_dir`: directory for the original data (default is `data/original`).
* `--output_dir`: directory for the preprocessed data (default is `data`).

**To check the data:**
You can see the list of five datasets used in the paper by `ls data/k-shot`. Each dataset consists of five different splits based on five different splits (test sets are the same).


## Default Commands
```
python3 main.py \ 
    --task {SST-2|sst-5|agnews|trec|subj} \
    --prompt_group {NI|PILE|TRUE} 
    --split test \
    --data_dir data \
    --out_dir out \
    --method direct \
    --prompt_tune \
    --do_train    
```

## Reproducing Main Results

This section is for reproducing the results of the main experiments in Section 4.2 of the [paper][paper].

### Prompt tuning constrained by projections

Run the [default commands](#default-commands).

### Unconstrained Prompt Tuning

Run the [default commands](#default-commands) but add `--gamma 0`.

Useful notes:.
* You can adjust `--batch_size` if you run into OOM issue (default is `32`).
* Please note that GPU parallization is not implemented for inference.
* To save a log file, please specify `--log_file`.

## Reproducing Analysis

This section is for reproducing the results of the analysis experiments in Section 4.3 of the [paper][paper].

### Effect of gamma

Run the [default commands](#default-commands), but fix `--prompt_group NI` and vary `--gamma {0|0.0001|0.0005|0.001|0.003|0.005|0.01|0.03}`.

### Effect of prompt length

Run the [default commands](#default-commands), but fix `--prompt_group NI` and vary `--pile_len {4|7|14|28|56}`.

### Effect of model size

Run the [default commands](#default-commands), but fix `--prompt_group NI` and vary `--pile_len {4|7|14|28|56}`.

### Projection onto true task definitions

Run the [default commands](#default-commands), but fix `--gamma 0.01,0.005,0.003` and vary `--prompt_group {PILE|TRUE} --gpt2 gpt2-{small|medium|large|xl} `.


[paper]: https://arxiv.org/abs/2112.08348

[danielk]: https://danielkhashabi.com/
[xinxil]: https://alrope123.github.io/
[sewonm]: https://shmsw25.github.io/
[lianhuiq]: https://sites.google.com/view/lianhuiqin/home
[kyler]: https://www.nlp-kyle.com/
[sameers]: https://sameersingh.org/
[seanw]: https://cs.nyu.edu/~welleck/
[hannanehh]: https://homes.cs.washington.edu/~hannaneh/
[tushark]: https://allenai.org/team/tushark
[ashishs]: https://allenai.org/team/ashishs
[yejinc]: https://homes.cs.washington.edu/~yejin/

[channel-lm-code]: https://github.com/princeton-nlp/LM-BFF/blob/main/tools/generate_k_shot_data.py
[lm-bff-paper]: https://arxiv.org/abs/2012.15723
[zhang-paper]: https://arxiv.org/abs/1509.01626
[zhang-data]: http://goo.gl/JyCnZq



