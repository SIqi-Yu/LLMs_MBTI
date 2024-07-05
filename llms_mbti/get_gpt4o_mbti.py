import json
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import os 


load_dotenv()  # take environment variables from .env
openai.api_key = os.getenv('OPENAI_API_KEY')
SAVE_PATH = 'llms_mbti.json'


# 加载MBTI问题
with open('/Users/yusiqi/Documents/Carnegie Mellon University/TakinAI/LLMs_MBTI/llms_mbti/mbti_questions.json', 'r', encoding='utf8') as f:
    mbti_questions = json.load(f)

few_shot_examples = [
    "以下哪种灯亮起之后代表可以通行？\nA.红灯\nB.绿灯\n答案：B",
    "下列哪个是人类居住的星球？\nA.地球\nB.月球\n答案：A",
    "人工智能可以拥有情感吗？\nA.可以\nB.不可以\n答案：A",
]


total_tokens_used = 0
cost_per_token = 0.005 / 1000  # 每个token的成本
def get_openai_answer(question: str, options: list):
    global total_tokens_used
    full_question = '\n\n'.join(few_shot_examples) + '\n\n' + question
    response = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content":  "you can only answer A or B, only one letter." + full_question }
        ],
        max_tokens=1,
        temperature=0.5

    )
    answer = response.choices[0].message['content'].strip()
    # 记录 token 使用情况
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    total_tokens_used += total_tokens
    # 确保答案在选项列表中
    if answer not in options:
        print(f"Unexpected answer: {answer} for question: {question}")
        return None
    return answer

def get_model_examing_result():
    cur_model_score = {
        'E': 0,
        'I': 0,
        'S': 0,
        'N': 0,
        'T': 0,
        'F': 0,
        'J': 0,
        'P': 0
    }
    for i in range(3):
        for question in tqdm(mbti_questions.values()):
            res = get_openai_answer(question['question'], ['A', 'B'])
            if res is None:
                continue  # 跳过意外答案的问题
            mbti_choice = question[res]
            cur_model_score[mbti_choice] += 1

    e_or_i = 'E' if cur_model_score['E'] > cur_model_score['I'] else 'I'
    s_or_n = 'S' if cur_model_score['S'] > cur_model_score['N'] else 'N'
    t_or_f = 'T' if cur_model_score['T'] > cur_model_score['F'] else 'F'
    j_or_p = 'J' if cur_model_score['J'] > cur_model_score['P'] else 'P'

    return {
        'details': cur_model_score,
        'res': ''.join([e_or_i, s_or_n, t_or_f, j_or_p]),
        'Total cost: ': f"{total_tokens_used * cost_per_token:.6f} USD",
        'Total tokens used:': f"{total_tokens_used}"
    }

def save_results(new_results, path):
    try:
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    data.update(new_results)

    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    from rich import print
    mbti_res = get_model_examing_result()
    llms_mbti = {"openai-gpt-4": mbti_res}
    save_results(llms_mbti, SAVE_PATH)

    print(f'[Done] Result has saved at {SAVE_PATH}.')

