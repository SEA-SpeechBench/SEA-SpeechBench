import sacrebleu

def bleu(self, input_data):

    questions, references, predictions = input_data
    
    # Use sacrebleu.corpus_bleu for computing BLEU score
    bleu_score = sacrebleu.corpus_bleu(
        predictions, 
        [references],  # sacrebleu expects references as a list of lists
        tokenize='flores101'
    )

    results = {"bleu_score": bleu_score.score}
    
    # Create details list with sample-level information (following other metrics pattern)
    details = []
    for i, (question, reference, prediction) in enumerate(zip(questions, references, predictions)):
        sample_detail = {
            'question': question,
            'reference': reference,
            'model_prediction': prediction,
            'sample_index': i
        }
        details.append(sample_detail)

    return results, details