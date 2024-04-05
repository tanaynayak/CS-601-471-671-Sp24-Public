import openai
import random
from datasets import load_dataset

# Function to load and prepare the BoolQ dataset
def load_boolq_dataset():
    dataset = load_dataset("google/boolq")
    train_dataset = dataset['train']
    boolq_instances = [(item['question'], 'yes' if item['answer'] else 'no') for item in train_dataset]
    return boolq_instances

# Function to randomly select and interleave BoolQ instances for balanced prompts
def prepare_balanced_prompts(instances, num_yes=4, num_no=4):
    yes_instances = [inst for inst in instances if inst[1] == 'yes']
    no_instances = [inst for inst in instances if inst[1] == 'no']
    selected_yes = random.sample(yes_instances, num_yes)
    selected_no = random.sample(no_instances, num_no)
    interleaved = []
    for i in range(max(num_yes, num_no)):
        if i < len(selected_yes): interleaved.append(selected_yes[i])
        if i < len(selected_no): interleaved.append(selected_no[i])
    return interleaved

# Function to format instances into a GPT prompt
def format_as_prompt(instances):
    return "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in instances])

# Function to query the OpenAI API with a prompt
def query_openai(prompt, question):
    full_prompt = f"{prompt}\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=full_prompt,
        max_tokens=5,
        temperature=0.7,
    )
    return response.choices[0].text.strip().lower()

# Function to evaluate the model on BoolQ instances
def evaluate_model(boolq_instances, num_samples=30):
    evaluation_instances = random.sample(boolq_instances, num_samples)
    balanced_prompts = prepare_balanced_prompts(boolq_instances)
    prompt_text = format_as_prompt(balanced_prompts)
    correct_answers = 0

    for question, correct_answer in evaluation_instances:
        predicted_answer = query_openai(prompt_text, question)
        if predicted_answer == correct_answer:
            correct_answers += 1

    return correct_answers / num_samples

if __name__ == "__main__":
    openai.api_key = ''  # Replace with your API key if need testing
    boolq_instances = load_boolq_dataset()
    accuracy = evaluate_model(boolq_instances)
    print(f"Evaluation Accuracy: {accuracy:.2%}")
