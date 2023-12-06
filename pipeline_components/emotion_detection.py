def build_emotions_baseline_prompt_step1(text):
    """
    :input: test Reddit post (for inference)
    :output: zero-shot prompt for step 1 (eliciting emotions)
    """
    return f"""[Text] {text}
    [Emotions] fear, trust, joy, anticipation, anger, disgust, sadness
    [Question] From the text of a narrator describing a situation, please select emotions the narrator from the above emotions. Select only from the set provided. If none are applicable, output N/A.""" 

def build_emotions_baseline_prompt_step2(step1_output):
    """
    :input: test Reddit post (for inference), step1 output
    :output: zero-shot prompt for step 2 (eliciting emotions)
    """
    return f"""[Relevant Emotions] {step1_output}
    [Question] For each of the relevant emotions you selected above, please give an intensity rating of each emotion on a scale of low, medium, or high. If N/A was selected, output N/A again.""" 

def build_emotions_cot_prompt_step1(text, emotion):
    """
    :input: test Reddit post (for inference)
    :output: zero-shot prompt for step 1 (eliciting emotions)
    """
    return f"""[Text] {text}
    [Question] From the text of a narrator describing a situation, please determine, yes or no, whether the user is feeling {emotion}."""

def build_emotions_cot_prompt_step2(emotion):
    """
    :input: test Reddit post (for inference)
    :output: zero-shot prompt for step 1 (eliciting emotions)
    """
    return f"""If the answer to the last question is yes, please give an intensity rating of {emotion} on a scale of low, medium, or high. If the answer was no, please output N/A."""

def build_emotions_iterative_prompt_step1(text, appraisal_analysis, prior_output):
    """ 
    :input: test Reddit post (for inference)
    :output: zero-shot prompt for step 1 (eliciting emotions)
    """
    if(appraisal_analysis == ""):
        return f"""[Text] {text}
        [Emotions] fear, trust, joy, anticipation, anger, disgust, sadness
        [Prior Output] {prior_output}
        [Question] From the text of a narrator describing a situation, please select emotions the narrator from the above emotions. Please revise the prior selected emotions based on the appraisal analysis, using the above set only. If none are applicable, output N/A."""
    else:
        return f"""[Text] {text}
        [Appraisal Analysis] {appraisal_analysis}
        [Emotions] fear, trust, joy, anticipation, anger, disgust, sadness
        [Prior Output] {prior_output}
        [Question] From the text of a narrator describing a situation, please select emotions the narrator from the above emotions. Please revise the prior selected emotions based on the appraisal analysis, using the above set only. If none are applicable, output N/A."""

def build_emotions_iterative_prompt_step2(text, appraisal_analysis, step1_output):
    """
    :input: test Reddit post (for inference), step1 output
    :output: zero-shot prompt for step 2 (eliciting emotions)
    """
    if(appraisal_analysis == ""):
        return f"""[Text] {text}
        [Relevant Emotions] {step1_output}
        [Question] For each of the relevant emotions you selected above, based on the text and appraisal analysis, please give an intensity rating of each emotion on a scale of low, medium, or high. If N/A was selected, output N/A again.""" 
    else:
        return f"""[Text] {text}
        [Appraisal Analysis] {appraisal_analysis}
        [Relevant Emotions] {step1_output}
        [Question] For each of the relevant emotions you selected above, based on the text and appraisal analysis, please give an intensity rating of each emotion on a scale of low, medium, or high. If N/A was selected, output N/A again.""" 
    