revision_prompt = """please revise the reappraisal response to additionally address this feedback, while minimally modifying the original response"""

def build_iterative_step_baseline(post, prev_step):
    return f"""[Text] {post}\n\n[Reappraisal Response] {prev_step}\n[Feedback]  Please revise the reappraisal response to help the narrator reappraise the situation better. Your response should be concise and brief."""

def build_iterative_step_baseline_guideline(post, guidance, prev_step):
    return f"""[Text] {post}\n\n[Reappraisal Response] {prev_step}\n[Feedback] {guidance} Taking this into account, {revision_prompt}. Your response should be concise and brief."""

def build_iterative_step_w_appraisal(post, appraisal, prev_step):
    return f"""[Text] {post}\n\n[Reappraisal Response] {prev_step}\n[Feedback] {appraisal}. Based on the above appraisal, {revision_prompt}. Your response should be concise and brief."""

def build_emotional_support_prompt(post, prev_step):
    return f"""[Text] {post}\n\n[Reappraisal Response] {prev_step}\n[Feedback]  Please integrate the reappraisal response into an emotional support response, with the aim of making the narrator feel better about the situation. Your emotional support response should be specific about the reappraisal, and include personal examples from the reappraisal that the narrator can relate to. Talk to the narrator as if you were a close friend. Your response should be concise and brief."""