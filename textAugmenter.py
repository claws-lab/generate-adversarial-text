import textattack as ta
from textattack.constraints.pre_transformation import RepeatModification
from textattack.constraints.pre_transformation import StopwordModification
from textattack.augmentation import Augmenter
from datasets import load_dataset
import pandas as pd

# Set up transformations
Transformations = [
    # Sentence Transformation
    ta.transformations.BackTranslation,

    # Word Insertion Transformation
    ta.transformations.WordInsertionMaskedLM,
    ta.transformations.WordInsertionRandomSynonym,
    ta.transformations.WordMergeMaskedLM,

    # Word Swap Transformation
    ta.transformations.WordSwapChangeLocation,
    ta.transformations.WordSwapChangeName,
    ta.transformations.WordSwapChangeNumber,
    ta.transformations.WordSwapContract,
    ta.transformations.WordSwapEmbedding,
    ta.transformations.WordSwapExtend,

    # ta.transformations.WordSwapGradientBased, # Model based: Requires white box access
    ta.transformations.WordSwapHomoglyphSwap,
    ta.transformations.WordSwapHowNet,
    ta.transformations.WordSwapInflections,
    ta.transformations.WordSwapMaskedLM,
    ta.transformations.WordSwapNeighboringCharacterSwap,
    ta.transformations.WordSwapQWERTY,
    ta.transformations.WordSwapRandomCharacterDeletion,
    ta.transformations.WordSwapRandomCharacterInsertion,
    ta.transformations.WordSwapRandomCharacterSubstitution,
    ta.transformations.WordSwapWordNet,

    # Word Deletion Transformation
    ta.transformations.WordDeletion,
    ta.transformations.WordInnerSwapRandom,
]

# Set up Composite Transformations
Prebuilt_Augmenters = [
    # Composite attacks, changes in constraints or parameters
    ta.augmentation.BackTranslationAugmenter,
    ta.augmentation.CLAREAugmenter,
    ta.augmentation.CharSwapAugmenter,
    ta.augmentation.CheckListAugmenter,
    ta.augmentation.EasyDataAugmenter,
    ta.augmentation.EmbeddingAugmenter,

    # Already included in the Transformations list
    # ta.augmentation.DeletionAugmenter,
    # ta.augmentation.WordNetAugmenter
]

def load_data():
    """
    Load the text data for augmentation.
    Returns:
        text (list): List of text data.
    """
    data = load_dataset("PolyAI/banking77")
    text = data['train']['text'][0:10]
    return text

def apply_augmentations(text):
    """
    Apply the defined augmentations and composite augmentations to the text.
    Args:
        text (list): List of text data.
    Returns:
        result (list): List of augmented text data.
    """
    result = []

    # Set up constraints
    constraints = [RepeatModification(), StopwordModification()]

    for transformation in Transformations:
        _type = transformation.__name__
        augmenter = Augmenter(
            transformation=transformation(),
            constraints=constraints,
            pct_words_to_swap=0.1, 
            transformations_per_example=1
        )

        for t in text:
            try:
                res = augmenter.augment(t)[0]
                if res != t:
                    result.append([_type, t, res])
            except:
                pass

    for augmentation in Prebuilt_Augmenters:
        _type = augmentation.__name__
        augmenter = augmentation()

        for t in text:
            try:
                res = augmenter.augment(t)[0]
                if res != t:
                    result.append([_type, t, res])
            except:
                pass

    return result

if __name__ == "__main__":
    text = load_data()
    result = apply_augmentations(text)

    # Convert list of lists to pandas DataFrame
    df = pd.DataFrame(result, columns=['Type', 'Original', 'Augmented'])
    df.to_csv('augmented.csv', index=False)
