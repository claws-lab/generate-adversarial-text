import ssl 
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import textattack as ta
import OpenAttack as oa
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# List of prebuilt TextAttack attacks
Prebuilt_TextAttacks = [
    # ta.attack_recipes.a2t_yoo_2021.A2TYoo2021, # Error in the library 
    # ta.attack_recipes.faster_genetic_algorithm_jia_2019.FasterGeneticAlgorithmJia2019,
    # ta.attack_recipes.bae_garg_2019.BAEGarg2019, # High memory usage
    ta.attack_recipes.bert_attack_li_2020.BERTAttackLi2020,
    ta.attack_recipes.checklist_ribeiro_2020.CheckList2020,
    ta.attack_recipes.deepwordbug_gao_2018.DeepWordBugGao2018,
    # ta.attack_recipes.hotflip_ebrahimi_2017.HotFlipEbrahimi2017,
    ta.attack_recipes.input_reduction_feng_2018.InputReductionFeng2018,
    ta.attack_recipes.pso_zang_2020.PSOZang2020,
    ta.attack_recipes.pwws_ren_2019.PWWSRen2019,
    ta.attack_recipes.textfooler_jin_2019.TextFoolerJin2019,
    ta.attack_recipes.textbugger_li_2018.TextBuggerLi2018,
    ta.attack_recipes.pruthi_2019.Pruthi2019,
    ta.attack_recipes.clare_li_2020.CLARE2020
]
 
# List of prebuilt OpenAttack attacks
Prebuilt_OpenAttack = [
    # oa.attackers.BAEAttacker, # Available in TextAttack # High memory usage
    # oa.attackers.BERTAttacker, # Available in TextAttack
    # oa.attackers.DeepWordBugAttacker, # Available in TextAttack
    oa.attackers.FDAttacker,
    # oa.attackers.GANAttacker, # Poor Output Quality
    # oa.attackers.GEOAttacker, # Undefined
    oa.attackers.GeneticAttacker,
    oa.attackers.HotFlipAttacker, # Available in TextAttack
    # oa.attackers.PSOAttacker, # Available in TextAttack
    # oa.attackers.PWWSAttacker, # Available in TextAttack
    oa.attackers.SCPNAttacker,
    # oa.attackers.TextBuggerAttacker, # Available in TextAttack
    # oa.attackers.TextFoolerAttacker, # Available in TextAttack
    oa.attackers.UATAttacker,
    oa.attackers.VIPERAttacker
]

def load_data_TextAttack():
    """
    Load the dataset for TextAttack.

    Returns:
    - tokenized_dataset: The tokenized dataset for TextAttack.
    """
    dataset = load_dataset("PolyAI/banking77")
    tokenized_dataset = ta.datasets.HuggingFaceDataset(dataset["test"].shuffle(seed=0).select(range(100)))
    return tokenized_dataset

def load_data_OpenAttack():
    """
    Load the dataset for OpenAttack.

    Returns:
    - dataset: The dataset for OpenAttack.
    """
    dataset = load_dataset("PolyAI/banking77")
    dataset = dataset["test"].shuffle(seed=0).select(range(100))

    def dataset_mapping(x):
        return {
            "x": x["text"],
            "y": x["label"],
        }

    dataset = dataset.map(function=dataset_mapping, remove_columns=["text", "label"])
    return dataset

def load_model_TextAttack():
    """
    Load the TextAttack model for sequence classification.

    Returns:
    - model_wrapped: The wrapped HuggingFace model for TextAttack.
    """
    model_id = 'philschmid/BERT-Banking77'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model_wrapped = ta.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    return model_wrapped

def load_model_OpenAttack():
    """
    Load the OpenAttack model for sequence classification.

    Returns:
    - model_wrapped: The wrapped HuggingFace model for OpenAttack.
    """
    model_id = 'philschmid/BERT-Banking77'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model_wrapped = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)
    return model_wrapped


def apply_TextAttacks(model_wrapped, tokenized_dataset):
    """
    Apply prebuilt TextAttack attacks to the dataset.

    Args:
    - model_wrapped: The wrapped HuggingFace model for TextAttack.
    - tokenized_dataset: The tokenized dataset for TextAttack.

    Returns:
    - attacked: A list of attacked examples, including the attack type, original text, and perturbed text.
    """
    attacked = []
    for _attack in Prebuilt_TextAttacks:
        attack_module = _attack.__module__
        _type = attack_module.split('.')[-1]

        attack_args = ta.AttackArgs(
        num_examples=100,
        parallel=True,
        checkpoint_dir=None,
        disable_stdout=False
        )

        attack = _attack.build(model_wrapped)
        attacker = ta.Attacker(attack, tokenized_dataset, attack_args)
        results = attacker.attack_dataset()

        for res in results:
            # Wrongly classified examples are skipped by the textAttack library
            if 'SKIPPED' not in res.goal_function_result_str() and 'FAILED' not in res.goal_function_result_str():
                attacked.append([_type, res.original_text(), res.perturbed_text(), res.original_result.ground_truth_output])
    
    return attacked

def apply_OpenAttacks(model_wrapped, dataset):
    """
    Apply prebuilt OpenAttack attacks to the dataset.

    Args:
    - model_wrapped: The wrapped HuggingFace model for OpenAttack.
    - dataset: The dataset for OpenAttack.

    Returns:
    - attacked: A list of attacked examples, including the attack type, original text, and perturbed text.
    """
    attacked = []
    for _attack in Prebuilt_OpenAttack:
        _type = _attack.__name__

        attacker = _attack()
        attack_eval = oa.AttackEval(attacker, model_wrapped)
        result = attack_eval.ieval(dataset)
        
        for res in result:
            if(res['success'] == True):
                attacked.append([_type, res['data']['x'], res['result'], res['data']['y']])
    
    return attacked

if __name__ == "__main__":
    # Load TextAttack model and dataset
    model_wrapped = load_model_TextAttack()
    tokenized_dataset = load_data_TextAttack()

    # Apply TextAttack attacks
    attacked_TextAttack = apply_TextAttacks(model_wrapped, tokenized_dataset)

    # Load OpenAttack model and dataset
    model_wrapped = load_model_OpenAttack()
    dataset = load_data_OpenAttack()

    # Apply OpenAttack attacks
    attacked_OpenAttack = apply_OpenAttacks(model_wrapped, dataset)

    # Merge attacked examples and save to CSV
    merge_attacked = attacked_TextAttack + attacked_OpenAttack
    df = pd.DataFrame(merge_attacked, columns=['Type', 'Original', 'Attacked', 'Original Label'])
    df.to_csv('attacked.csv', index=False)
