#==============#
# Importations #
#==============#
import os
import ast
import sys
import torch
import random
import numpy as np
import pandas as pd
from absl import flags

from tqdm import tqdm
from pprint import pprint
from transformers import set_seed
from IPython.display import display
from nltk.tokenize import word_tokenize
from transformers import (AutoTokenizer, AutoModelForCausalLM)

from pipeline_components.chat_agents import (
    OpenAIAgent,
    LLaMA2ChatAgent,
    MistralChatAgent
)

from pipeline_components import emotion_detection
from openai import AzureOpenAI, OpenAI

# openai_api_key = open("./auth_keys/openai_key.txt", "r").read()
# os.environ["OPENAI_API_KEY"] = openai_api_key

huggingface_auth_key = open("./auth_keys/huggingface_auth_key.txt", "r").read()

AZURE_OPENAI_KEY = open("./auth_keys/azure_openai_key.txt", "r").read()
AZURE_OPENAI_ENDPOINT = open("./auth_keys/azure_endpoint.txt", "r").read()
os.environ["AZURE_OPENAI_KEY"] = AZURE_OPENAI_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT

FLAGS = flags.FLAGS

# Define the LLM
flags.DEFINE_string("model", "gpt-4-turbo", "Specify the LLM.")

# Hyper-parameters for LLMs
flags.DEFINE_float("temperature", 0.1, "The value used to modulate the next token probabilities.")
flags.DEFINE_integer("max_tokens", 512, "Setting max tokens to generate.")
flags.DEFINE_integer("seed", 36, "Setting seed for reproducible outputs.")

# Loading data
flags.DEFINE_string("data", "./CovidET_emotions/CovidET-ALL_w_appraisal.csv", "Path pointing to the directory containing the posts and emotion annotations.")
flags.DEFINE_string("appraisal_questions", "./prompts/appraisal_questions.txt", "File with all appraisal questions.")

# Define experiment mode
flags.DEFINE_string("experiment_mode", "baseline", "Specify experiment.")
flags.DEFINE_string("use_appraisals", 'n', "Add appraisals as context")

# Parse the flags
flags.FLAGS(sys.argv)

#==========#
# Set seed #
#==========#
set_seed(FLAGS.seed)
random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
torch.cuda.manual_seed_all(FLAGS.seed)


plutchik_wheel = {
    'na': ['na', 'na2', 'na3'],
    'fear': ['apprehension', 'fear', 'terror'],
    'trust': ['acceptance', 'trust', 'admiration'],
    'joy': ['serenity', 'joy', 'ecstasy'],
    'anticipation': ['interest', 'anticipation', 'vigilance'],
    'anger': ['annoyance', 'anger', 'rage'],
    'disgust': ['boredom', 'disgust', 'loathing'],
    'sadness': ['pensiveness', 'sadness', 'grief']
}

emotions_mapping = {emotion: id for emotion, id in zip(plutchik_wheel.keys(), range(0,8))}

if __name__ == "__main__":

    #===========================#
    # Establishing Class Agents #
    #===========================#
    if "Llama-2" in FLAGS.model:
        ChatAgent = LLaMA2ChatAgent(
            FLAGS,
            tokenizer = AutoTokenizer.from_pretrained(FLAGS.model, token=huggingface_auth_key),
            model = AutoModelForCausalLM.from_pretrained(FLAGS.model, device_map='auto', torch_dtype=torch.float16, token=huggingface_auth_key),
            new_system_prompt = """Respond with a response in the format requested by the user. Do not acknowledge my request with "sure" or in any other way besides going straight to the answer.""",
        )

    elif "gpt" in FLAGS.model:
        ChatAgent = OpenAIAgent(
            FLAGS,
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),  
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2023-05-15"
            ),
            new_system_prompt = """Respond with a response in the format requested by the user. Do not acknowledge my request with "sure" or in any other way besides going straight to the answer.""",
        )

    elif "mistral" in FLAGS.model:
        ChatAgent = MistralChatAgent(
            FLAGS,
            tokenizer = AutoTokenizer.from_pretrained(FLAGS.model),
            model = AutoModelForCausalLM.from_pretrained(FLAGS.model, device_map='auto', torch_dtype=torch.float16),
            new_system_prompt = """Respond with a response in the format requested by the user. Do not acknowledge my request with "sure" or in any other way besides going straight to the answer.""",
        )

    else:
        raise ValueError

    model_name = FLAGS.model.split('/')[-1]

    print(FLAGS.model)
    #==================#
    # Elicit responses #
    #==================#
    dimensions_df = pd.DataFrame(index = range(1, 25))
    dim_files = [FLAGS.appraisal_questions]

    for dim_file_name in dim_files:
        with open(dim_file_name) as dim_file:
            raw_txt = dim_file.read()
            if 'dim_name' not in dimensions_df.columns:
                dimensions_df['dim_name'] = pd.Series(raw_txt.split('\n'), index = range(1, 25)).apply(lambda line: line[:line.find('=') - 1])
            dimensions_df[dim_file_name[10:-4]] = pd.Series(raw_txt.split('\n'), index = range(1, 25)).apply(lambda line: line[line.find('=') + 1:])
    #display(dimensions_df)

    eval_set = pd.read_csv(FLAGS.data).head(1)

    path = f"""llm_output/{FLAGS.experiment_mode}/{"baseline" if FLAGS.use_appraisals == 'n' else "appraisal"}"""
    if not os.path.exists(path):
        os.makedirs(path)

    csv_file_path = f"{path}/{model_name}.csv"

    if os.path.exists(csv_file_path):
        print (f"Loading responses from {csv_file_path}")
        responses_df = pd.read_csv(csv_file_path)
        elicited_unique_ids = responses_df["id"].unique()
        eval_set = eval_set[~eval_set["id"].isin(elicited_unique_ids)]

    for _, row in tqdm(eval_set.iterrows(), total=len(eval_set)):
        post_responses = dict()

        post = row["post"]
        post_responses["post"] = post

        post_id = eval_set[eval_set["post"] == post]['id'].iloc[0]
        post_responses["id"] = eval_set[eval_set["post"] == post]['id'].iloc[0]

        if FLAGS.experiment_mode == "baseline":
            if FLAGS.use_appraisals == 'y':
                ChatAgent.append_history(row["all_natural_language_ratings"])

            step1_prompt = emotion_detection.build_emotions_baseline_prompt_step1(post)
            step1_output = ChatAgent.chat(step1_prompt)
            print(step1_output)

            step2_prompt = emotion_detection.build_emotions_baseline_prompt_step2(step1_output)
            step2_output = ChatAgent.chat(step2_prompt)
            print(step2_output)
            ChatAgent.reset()

            post_responses["llm_emotions_raw"] = step1_output
            post_responses["llm_intensities_raw"] = step2_output
        
        elif FLAGS.experiment_mode == "cot":
            if FLAGS.use_appraisals == 'y':
                ChatAgent.append_history(row["all_natural_language_ratings"])
                
            for emotion in emotions_mapping:
                step1_prompt = emotion_detection.build_emotions_cot_prompt_step1(post, emotion)
                step1_output = ChatAgent.chat(step1_prompt)
                print(step1_output)
        
                step2_prompt = emotion_detection.build_emotions_cot_prompt_step2(emotion)
                step2_output = ChatAgent.chat(step2_prompt)
                print(step2_output)

                post_responses[f"llm_{emotion}_raw"] = step1_output
                post_responses[f"llm_{emotion}_intensity_raw"] = step2_output
                ChatAgent.reset()

        elif FLAGS.experiment_mode == "self_refine":
            step1_prompt = emotion_detection.build_emotions_baseline_prompt_step1(post)
            step1_output = ChatAgent.chat(step1_prompt)
            print(step1_output)

            step2_prompt = emotion_detection.build_emotions_baseline_prompt_step2(step1_output)
            step2_output = ChatAgent.chat(step2_prompt)
            print(step2_output)
            ChatAgent.reset()

            post_responses["llm_emotions_initial_raw"] = step1_output
            post_responses["llm_intensities_initial_raw"] = step2_output
        
            for dim in range(1, 25):
                appraisal_combined = row[f"natural_language_dim{dim}"] + '\n' + row[f"dim{dim}_rationale"] if FLAGS.use_appraisals == 'y' else ""
                
                step1_prompt = emotion_detection.build_emotions_iterative_prompt_step1(post, appraisal_analysis = appraisal_combined, prior_output = step2_output)
                step1_output = ChatAgent.chat(step1_prompt)
                print(step1_output)

                step2_prompt = emotion_detection.build_emotions_baseline_prompt_step2(step1_output)
                step2_output = ChatAgent.chat(step2_prompt)
                print(step2_output)

                ChatAgent.reset()

                post_responses[f"llm_emotions_dim_{dim}_raw"] = step1_output
                post_responses[f"llm_intensities_dim_{dim}_raw"] = step2_output
                
        else: raise ValueError

        #df = pd.DataFrame([post_responses])
        #df.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))
        ChatAgent.reset()

sys.exit()