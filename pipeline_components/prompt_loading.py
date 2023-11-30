def build_appraisal_prompt(text, appraisal_q):
    """
    :input: test Reddit post (for inference), appraisal_q (for one dimension)
    :output: zero-shot prompt for step 1 (eliciting appraisals)
    """
    return f"""[Text] {text}

[Question] {appraisal_q} Please provide your answer in the following format: <likert>[]</likert><rationale>[]</rationale>. Your response should be concise and brief."""


def build_emotional_support_prompt(responses_dict):
    test_post = responses_dict['Reddit Post']
    reappraisals = ""
    for key in responses_dict.keys():
        if "reappraisal_question" in key:
            reappraisals += f"[Reappraisal Question] {responses_dict[key]}\n"
        elif "reappraisal_output" in key:
            reappraisals += f"[Reappraisal Response] {responses_dict[key]}\n\n"
        else: pass

    return f"""You will be presented with a text in which the narrator is describing a situation, together with reappraisals responses that aim to guide the narrator to view the situation from a different perspective. Please summarize all these reappraisal responses into an emotional support message with the goal to make the narrator feel better.

[Text] {test_post}

{reappraisals}

[Question] Please summarize all the reappraisal responses into an emotional support response, with the aim of making the narrator feel better about the situation. Your summary should be specific about each reappraisal, and include personal examples from the reappraisals that the narrator can relate to. Please organize your summary in a coherent manner. Talk to the narrator as if you were a close friend. Your response should be concise and brief."""


def build_emotional_support_prompt_baseline(responses_dict):
    """
    :Aim: to elicit the most basic emotional responses from LLMs, as contrast to the ones using reappraisals, to see how pristine and templated the answers are.
    """
    test_post = responses_dict['Reddit Post']

    return f"""You will be presented with a text in which the narrator is describing a situation. Please come up with a response to make the narrator feel better.

[Text] {test_post}

[Question] Please come up with a response to make the narrator feel better. Your response should be concise and brief."""