## What is this

An example repo for using [Zelda Rose](https://zeldarose.readthedocs.io) to train/fine tune a next token prediction (aka causal lm) transformer model. It includes data from the [Diplomacy](https://github.com/DenisPeskoff/2020_acl_diplomacy/) and [CORAAL](https://oraal.github.io/coraal) corpora. I'll figure out the exact licence later, but assume CC-BY 4.0.

## How to run this

```
uv pip install -U -r requirements.lst
zeldarose transformer --config ntp-config.toml  --strategy ddp_find_unused_parameters_true --accelerator gpu --num-devices 2 --tokenizer "EleutherAI/pythia-160m-deduped" --pretrained-model "EleutherAI/pythia-160m-deduped"  --out-dir local/train-out --model-name "pythia-160m-deduped+diplomacy" --max-epochs 8 --device-batch-size 8 --precision 32-true --tf32-mode medium --val-data diplomacy/validation.txt  --num-workers 6 --step-save-period 4096 diplomacy/train.txt
```

- The directory you gave to `--out-dir` will have the final and partway models and tf event file that you can peruse using tensorboard.

## How to play with this

- Tuning parameters are in `ntp-config.toml`
  - See <https://zeldarose.readthedocs.io/en/latest/training_configurations.html> for what they mean
- Parameters in the CLI like `--precision`, see `zeldarose --help`
- Use your own data files obviously, the format is [raw text](https://zeldarose.readthedocs.io/en/latest/tasks/mlm.html#inputs-and-outputs)


## Does it work?

Looks like it?

```python
>>> from transformers import pipeline
>>> generator = pipeline(model="local/train-out/pythia-160m-deduped+diplomacy", task="text-generation")
Device set to use cuda:0
>>> generator("I can't believe you did such a", do_sample=False)
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
[{'generated_text': "I can't believe you did such a great turn."}]
>>> generator_orig = pipeline(model="EleutherAI/pythia-160m-deduped", task="text-generation")
Device set to use cuda:0
>>> generator_orig("I can't believe you did such a", do_sample=False)
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
[{'generated_text': 'I can\'t believe you did such a thing." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry." "I\'m sorry'}]
```

Also numbers go up (or down, but improving in any case):

<img width="1146" height="971" alt="A screenshot of tensoboard, showing that train and validation accuracies go up and loss and perplexities go down while training." src="https://github.com/user-attachments/assets/279e252e-0849-4de9-909f-f71214d58600" />

