import json
import random
import re

input_file = 'fine_tuning_dataset.jsonl'
output_file = 'natural_golden_dataset.json'

def clean_title(title):
    # Remove common prefixes/suffixes in issue titles
    title = re.sub(r'\[.*?\]', '', title) # Remove [Bug], [Feature], etc.
    title = re.sub(r'\(.*?\)', '', title) # Remove (refs #123)
    title = re.sub(r'^(Bug|Issue|Feature|Error):\s*', '', title, flags=re.IGNORECASE)
    title = title.strip()
    return title

def humanize_question(title, repo):
    templates = [
        f"I'm running into an issue with {repo}: {title}. How can I fix this?",
        f"Has anyone seen '{title}' in {repo}? I can't figure out the cause.",
        f"What is the resolution for '{title}'?",
        f"I'm getting an error: {title}. Is there a known workaround?",
        f"In {repo}, {title}. Any ideas why?",
        f"Help needed with {repo}. {title}.",
        f"I'm seeing '{title}'. Is this a known bug in {repo}?",
        f"Could you explain how to resolve '{title}'?"
    ]
    return random.choice(templates)

def clean_answer(text):
    # Remove the standard intro line
    text = text.replace("The issue was addressed with the following discussion:", "")
    
    # Remove "User '...' said:" lines
    text = re.sub(r"User '.*?' said:", "", text)
    
    # Remove dividers
    text = text.replace("---", "")
    
    # Remove "<details>...</details>" blocks which are often noisy in these logs
    text = re.sub(r'<details>.*?</details>', '', text, flags=re.DOTALL)
    
    # Remove triage/bot boilerplates
    text = re.sub(r'This issue is currently awaiting triage\..*?(\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'/sig .*?(\n|$)', '', text)
    text = re.sub(r'/area .*?(\n|$)', '', text)
    text = re.sub(r'/assign.*?(\n|$)', '', text)
    text = re.sub(r'/close.*?(\n|$)', '', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text

def extract_repo(text):
    match = re.search(r"in the '(.*?)' repository", text)
    return match.group(1) if match else "the repository"

def extract_title(text):
    match = re.search(r"titled '(.*?)'", text)
    return match.group(1) if match else "Unknown Issue"

def determine_category(text):
    text = text.lower()
    if any(x in text for x in ['error', 'fail', 'crash', 'bug', 'panic', 'broken', 'exception']):
        return 'troubleshooting'
    elif any(x in text for x in ['add', 'feature', 'support', 'request', 'new']):
        return 'feature_request'
    elif any(x in text for x in ['doc', 'manual', 'guide']):
        return 'documentation'
    elif 'security' in text or 'vulnerability' in text:
        return 'security'
    else:
        return 'general'

data = []
try:
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Shuffle and pick 75
    selected_lines = random.sample(lines, min(75, len(lines)))

    golden_data = []

    for i, line in enumerate(selected_lines):
        try:
            entry = json.loads(line)
            instruction = entry.get('instruction', '')
            response = entry.get('response', '')
            
            repo = extract_repo(instruction)
            raw_title = extract_title(instruction)
            cleaned_title = clean_title(raw_title)
            category = determine_category(instruction + " " + raw_title)
            
            # Context remains the "Source Document" content
            # combining instruction (description) + response (resolution)
            context_text = f"Repository: {repo}\nTitle: {raw_title}\n\n{instruction}\n\nResolution/Discussion:\n{response}"
            
            # Humanized Question
            question_text = humanize_question(cleaned_title, repo)
            
            # Humanized Answer
            ideal_answer_text = clean_answer(response)
            
            # Skip if answer became empty after cleaning
            if not ideal_answer_text:
                continue

            golden_entry = {
                "id": f"gen_nat_{i:03d}",
                "category": category,
                "source_file": f"{repo.replace('/', '_')}_issue.json",
                "context": context_text,
                "question": question_text,
                "ideal_answer": ideal_answer_text
            }
            
            golden_data.append(golden_entry)
        except Exception as e:
            continue

    with open(output_file, 'w') as f:
        json.dump(golden_data, f, indent=2)

    print(f"Successfully created {output_file} with {len(golden_data)} entries.")
    # Print a sample to check "naturalness"
    print(json.dumps(golden_data[0], indent=2))

except FileNotFoundError:
    print("Error: fine_tuning_dataset.jsonl not found.")