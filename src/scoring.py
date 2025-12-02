import pandas as pd
import numpy as np
import json
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def clean_text(text):
    punctuations = '''!()-[]{};:'"\.,<>/?@#$%^&*_~'''
    for p in punctuations:
        text = text.replace(p, "")
    text = text.lower()
    return text

class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute the competition score by comparing predicted answers and explanations
    against the ground truth.

    The ID column specified by `row_id_column_name` is removed from both dataframes.
    For each row, the predicted answer is matched to the ground-truth answer using
    case-insensitive exact comparison. If the answer is correct, the BLEU score is
    computed between the predicted explanation and each reference explanation, and
    the highest BLEU value is kept. If the answer is incorrect, the BLEU score is 0.

    The final score is the mean BLEU score across all samples (including zeros).

    Parameters
    ----------
    solution : pd.DataFrame
        DataFrame with ground-truth `answer` and list-based `explanation` fields.
    submission : pd.DataFrame
        DataFrame with predicted `answer` and `explanation` fields.
    row_id_column_name : str
        Column name to remove before scoring.

    Returns
    -------
    float
        Mean conditional BLEU score across all samples.
    """
    # rewrite the eval_metrics function to use solution and submission dataframes
    del solution[row_id_column_name]
    # del submission[row_id_column_name]
    
    corrects = np.zeros(len(solution))
    sample_bleu = np.zeros(len(solution))

    assert len(solution) == len(submission), "Solution and submission must have the same number of rows."

    for i in range(len(solution)):
        sol_answer = solution.iloc[i]['answer']
        subm_answer = submission.iloc[i]['answer']
        sol_explanations = json.loads(solution.iloc[i]['explanation'].replace("'", '"'))
        subm_explanation = submission.iloc[i]['explanation']

        if str(sol_answer).strip().lower() == str(subm_answer).strip().lower():
            corrects[i] = 1
            # Compute BLEU score for explanations
            best_bleu = 0.0
            for sol_explanation in sol_explanations:
                reference = [clean_text(sol_explanation).split()]
                candidate = clean_text(subm_explanation).split()
                try:
                    curr_bleu = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
                    if curr_bleu > best_bleu:
                        best_bleu = curr_bleu
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Error computing BLEU score at index: {i}")
            sample_bleu[i] = best_bleu
        else:
            sample_bleu[i] = 0.0

    average_bleu = np.mean(sample_bleu)
    
    accuracy = np.mean(corrects)
    return accuracy, average_bleu

def submission_postprocess(submission: pd.DataFrame) -> pd.DataFrame:
    # extract the first work as the answer
    def extract_answer_last(answer: str) -> str:
        return answer.strip().split(' ')[-1].lower().replace('.', '')
    
    def extract_answer_first(answer: str) -> str:
        return answer.strip().split('.')[0].lower()
    
    submission['answer'] = submission['answer'].apply(extract_answer_last)
    # submission['answer'] = submission['answer'].apply(extract_answer_first)
    return submission


def main():
    # Example usage
    submission_root = "/home/jnlp/minhnt/AdvML/results/"
    # submission_file = "dev_dl/tbd_nets.csv"
    submission_file = "dev_standard_prompt/qwen2_5_vl_3b_lora_dev_output_hf.csv"
    submission_path = os.path.join(submission_root, submission_file)
    # submission_csv_data = pd.read_json(submission_path, lines=True)
    submission_csv_data = pd.read_csv(os.path.join(submission_root, submission_file))
    
    solution = pd.read_csv("/home/jnlp/minhnt/AdvML/custom_dataset/dev_split.csv")
    # submission = pd.read_csv("submission.csv")
    submission = submission_postprocess(submission_csv_data)
    print(submission["answer"][0])
    row_id_column_name = "id"

    accuracy, avg_bleu = score(solution, submission, row_id_column_name)
    print(submission_file)
    print(f"Accuracy: {accuracy}")
    print(f"Average BLEU Score: {avg_bleu}")
    
if __name__ == "__main__":
    main()