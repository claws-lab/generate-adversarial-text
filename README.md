# Text Augmenter

## Description:

The 'textAugmenter' applies a set of transformations (individual and composite) to the given dataset to generate augmented samples. These samples help generalize better by accounting for possible variations. The function uses all possible transformations available in the TextAttack Library to perform the augmentation. 

## Usage Example

1. Select the required transformations from the 'Transformations' and 'Prebuilt_Augmenters' list. 
2. Load the dataset to be augmented using the 'load_dataset()' function.
3. Run the script to generate the augmentations. The results will be stored in 'augmented.csv' file. 



# Text Attacker

## Description:

Text attacks are targeted augmentations aimed at fooling a given model. The augmentations are generally crafted to attack the most important words in the sentence. The generated attack examples can be used for evaluating the adversarial robustness of the given model. To cover a large set of attacks, we use TextAttack and the OpenAttack libraries. 

NOTES: 
1. There are overlaps between the attacks in the TextAttack and the OpenAttack library. We prefer the TextAttack library, and therefore ignore OpenAttack attacks that are already present in TextAttack (these attacks are mentioned in the comments). Also, due to issues in the library, missing file, and high memory requirements few other attacks are also commented out.  
2. The processed dataset for the TextAttack in 'load_data_TextAttack()' must be tokenized, and for OpenAttack the input text should be mapped to "x" and the label should be mapped to "y" in 'load_data_OpenAttack'. 
3. The models for both the libraries should be wrapped using their corresponding wrappers. We use HuggingFace models, for using other libraries refer:
TextAttack Wrappers: https://textattack.readthedocs.io/en/master/apidoc/textattack.models.wrappers.html
OpenAttack Wrappers: [TransformerClassifier] https://github.com/thunlp/OpenAttack/blob/master/OpenAttack/victim/classifiers/transformers.py
4. For TextAttacks, modify the 'attack_args' in the 'apply_TextAttacks()' function to change the (1) number of examples (num_examples should match the size of the dataset being passed to the function), (2) parallel execution, and (3) output storage. For more detailed information refer: https://textattack.readthedocs.io/en/master/api/attacker.html#textattack.AttackArgs
5. All the successful attacks from both the TextAttack and OpenAttack are merged, and stored in the 'attacked.csv' file. 



## Usage Example

1. Select the required attacks from the 'Prebuilt_TextAttacks' and 'Prebuilt_OpenAttack'.
2. Load the dataset for the TextAttack and OpenAttack in the 'load_data_TextAttack()' and 'load_data_OpenAttack()' functions. 
3. Load the models for the TextAttack and OpenAttack in the 'load_model_TextAttack()' and 'load_model_OpenAttack()' functions. 
4. The selected attacks are applied to the given model and dataset, and the successful results are stored in 'attacked.csv' file. 



