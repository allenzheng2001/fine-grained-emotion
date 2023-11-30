emotional_support_question = """Please synthesize a summary from all the reappraisal responses, with the aim to make the narrator feel better about the situation. Your summary should address each single reappraisal in a coherent manner, and make sure to include personal examples from the reappraisals that the narrator can relate to. Talk to the narrator as if you were a close friend. Your response should be concise and brief."""

def build_emotional_support_prompt(responses_dict):
    test_post = responses_dict['text']
    reappraisals = ""
    i = 1
    for key in responses_dict.keys():
        #if "reappraisal_question" in key:
            #reappraisals += f"[Reappraisal Question] {responses_dict[key]}\n"
        if "reappraisal_output" in key:
            reappraisals += f"[Reappraisal {i}] {responses_dict[key]}\n\n"
            i += 1
        else: pass

    return f"""You will be presented with a text in which the narrator is describing a situation, together with reappraisals responses that aim to guide the narrator to view the situation from a different perspective. Please summarize all these reappraisal responses into an emotional support message.


[Text] {test_post}


{reappraisals}


[Question] {emotional_support_question}"""


emotional_support_question_baseline = """Please provide an emotional support message with the aim to make the narrator feel better about the situation. Talk to the narrator as if you were a close friend. Your response should be concise and brief."""

def build_emotional_support_prompt_baseline(responses_dict):
    test_post = responses_dict['text']

    return f"""You will be presented with a text in which the narrator is describing a situation. Please provide an emotional support message to help the narrator feel better.


[Text] {test_post}

[Question] {emotional_support_question_baseline}"""