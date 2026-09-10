GENERAL_TEMPLATE = """\
    [Question]
    {question}

    [Reference]
    {reference}

    [Model Prediction]
    {prediction}

    [Task]
    Rate the model prediction based on its alignment with the reference, focusing on accuracy and relevance to the reference. Be critical.
    Score0: The prediction repeats or rephrases the question without giving an answer.
    Score0: The prediction is refusing to give concrete results, providing something like 'cannot decide'.
    Score0: The prediction is completely misaligned, providing incorrect or irrelevant information compared to the reference.
    Score1: The prediction shows minimal alignment, often misunderstanding or providing irrelevant details unrelated to the reference.
    Score2: The prediction recognizes the topic but diverges significantly from the reference in accuracy or relevance.
    Score3: The prediction aligns with the reference generally but lacks detail or precise accuracy in some aspects.
    Score4: The prediction is mostly accurate and relevant, closely following the reference but could be clearer or more detailed.
    Score5: The prediction is highly accurate, detailed, and matches the reference perfectly, capturing its essence and detail.

    Your response should be formatted as follows:
    Explanation: (Provide a concise explanation of your rating, comparing the reference with the model prediction. "The reference is [XXX], while the model prediction is [YYY]. I think ...")
    Rating: (int)"""


BINARY_TEMPLATE = """\
    [Question]
    {question}

    [Reference]
    {reference}

    [Model Prediction]
    {prediction}

    [Task]
    Rate the model prediction based on its alignment with the reference, focusing on accuracy and relevance to the reference. Be critical.
    Score0: The prediction repeats or rephrases the question without giving an answer.
    Score0: The prediction is refusing to give concrete results, providing something like 'cannot decide'.
    Score0: The prediction is wrong, providing incorrect or irrelevant information compared to the reference.
    Score1: The prediction is correct, capturing or covering the meaning from the reference.

    Your response should be formatted as follows:
    Explanation: (Provide a concise explanation of your rating, comparing the reference with the model prediction. "The reference is [XXX], while the model prediction is [YYY]. I think ...")
    Rating: (int)"""