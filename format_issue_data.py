import json
import sys

# --- Configuration ---

# The raw data file you created with the scraper
INPUT_FILE = 'issue_diagnosis_dataset.jsonl'

# The new file that will contain the formatted data for fine-tuning
OUTPUT_FILE = 'fine_tuning_dataset.jsonl'

# Set a minimum number of comments. Issues with fewer comments are often
# not very informative for training a diagnostic model.
MINIMUM_COMMENTS = 2

# Set a maximum length for the issue body to avoid excessively long prompts.
# This helps prevent errors during training.
MAX_BODY_LENGTH = 3000 # in characters

# --- End Configuration ---

def format_instruction(record):
    """Creates the 'instruction' part of the training example."""
    repo = record.get('repo', 'N/A')
    title = record.get('issue_title', 'No Title').strip()
    body = record.get('issue_body', '').strip()

    # Truncate the body if it's too long
    if len(body) > MAX_BODY_LENGTH:
        body = body[:MAX_BODY_LENGTH] + "\n... (truncated)"

    return (
        f"A user reported the following issue titled '{title}' "
        f"in the '{repo}' repository. Please provide a summary of "
        f"the discussion that led to its resolution.\n\n"
        f"ISSUE DESCRIPTION:\n{body}"
    )

def format_response(record):
    """Creates the 'response' part of the training example from comments."""
    comments = record.get('comments', [])
    
    if not comments:
        return "There was no discussion on this issue."

    # Format each comment into a conversational turn
    conversation_parts = []
    for comment in comments:
        author = comment.get('author', 'unknown_user')
        body = comment.get('body', '').strip()
        if body: # Only include non-empty comments
            conversation_parts.append(f"User '{author}' said:\n---\n{body}\n---")
    
    full_conversation = "\n\n".join(conversation_parts)
    
    return (
        "The issue was addressed with the following discussion:\n\n"
        f"{full_conversation}"
    )


def main():
    """Main function to process the raw data and create the fine-tuning dataset."""
    print(f"Starting transformation of '{INPUT_FILE}'...")
    
    processed_count = 0
    skipped_count = 0
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                try:
                    record = json.loads(line)
                    
                    # --- Filtering Logic ---
                    # We only want issues with a meaningful discussion.
                    if len(record.get('comments', [])) < MINIMUM_COMMENTS:
                        skipped_count += 1
                        continue
                    
                    # --- Formatting ---
                    instruction = format_instruction(record)
                    response = format_response(record)
                    
                    # Create the final JSON object for the fine-tuning dataset
                    fine_tuning_record = {
                        "instruction": instruction,
                        "response": response
                    }
                    
                    # Write the new record to the output file
                    outfile.write(json.dumps(fine_tuning_record) + '\n')
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    print(f"Warning: Skipping a malformed line in {INPUT_FILE}", file=sys.stderr)
                    skipped_count += 1
                    continue
    
    except FileNotFoundError:
        print(f"[FATAL ERROR] The input file '{INPUT_FILE}' was not found.", file=sys.stderr)
        print("Please make sure the scraped data file is in the same directory.", file=sys.stderr)
        sys.exit(1)
        
    print("\n--- Transformation Complete ---")
    print(f"Successfully processed and formatted: {processed_count} records.")
    print(f"Skipped (due to not meeting criteria): {skipped_count} records.")
    print(f"Your fine-tuning dataset is ready at: '{OUTPUT_FILE}'")


if __name__ == '__main__':
    main()
