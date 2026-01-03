import os
import requests
import json
import time
import sys
from datetime import datetime
from dotenv import load_dotenv

# --- Configuration ---

# IMPORTANT: To handle your GitHub Personal Access Token securely, this script
# uses a .env file.
#
# 1. Install the python-dotenv library:
#    pip install python-dotenv
#
# 2. Create a file named .env in the same directory as this script.
#
# 3. Add your token to the .env file like this:
#    GITHUB_TOKEN="your_personal_access_token_here"
#
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# **UPDATED CONFIGURATION**
# We now use a dictionary to define repo-specific labels.
TARGET_CONFIG = {
    'microsoft/vscode': {
        'labels': ['bug', 'question', 'help-wanted', 'bug-report']
    },
    'facebook/react': {
        'labels': ['Type: Bug', 'Type: Question', 'good first issue', 'help wanted']
    },
    'kubernetes/kubernetes': {
        'labels': ['kind/bug', 'kind/question', 'good first issue', 'help wanted']
    }
}


# Output file where the data will be saved
OUTPUT_FILE = 'issue_diagnosis_dataset.jsonl'

# Maximum number of issues to scrape per repository.
# Set to a high number for full scrape, or a low number for testing.
MAX_ISSUES_PER_REPO = 100

# --- End Configuration ---

def get_rate_limit_status(headers):
    """Extracts rate limit information from the API response headers."""
    remaining = int(headers.get('X-RateLimit-Remaining', 0))
    limit = int(headers.get('X-RateLimit-Limit', 0))
    reset_timestamp = int(headers.get('X-RateLimit-Reset', 0))
    reset_time = datetime.fromtimestamp(reset_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return remaining, limit, reset_time

def respectful_wait(headers):
    """Waits if the rate limit is getting low."""
    remaining, limit, reset_time = get_rate_limit_status(headers)
    if remaining < 20: # Be conservative
        wait_time = max(datetime.fromtimestamp(reset_time) - datetime.now(), 0).total_seconds() + 10
        print(f"\n[RATE LIMIT] Low on requests ({remaining}/{limit}). Waiting for {int(wait_time)} seconds until {reset_time}...")
        time.sleep(wait_time)

def get_issue_comments(comments_url, headers):
    """Fetches all comments for a given issue."""
    try:
        response = requests.get(comments_url, headers=headers)
        response.raise_for_status()
        respectful_wait(response.headers)
        comments = response.json()
        
        # Format comments into a simpler structure
        formatted_comments = [
            {
                "author": comment['user']['login'],
                "body": comment['body']
            }
            for comment in comments if comment['body'] # Ensure comment body is not empty
        ]
        return formatted_comments
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Could not fetch comments from {comments_url}: {e}", file=sys.stderr)
        return []

def scrape_repo(repo_name, labels_to_search, headers):
    """Scrapes a single repository for closed issues with specific labels."""
    print(f"\n--- Starting scrape for repository: {repo_name} ---")
    
    # ** THE FIX IS HERE **
    # This logic correctly quotes each individual label IF it contains a space.
    # This is required for the GitHub API search to work correctly.
    quoted_labels = []
    for label in labels_to_search:
        if ' ' in label:
            quoted_labels.append(f'"{label}"')
        else:
            quoted_labels.append(label)
    
    labels_for_query = ','.join(quoted_labels)
    labels_query = f'label:{labels_for_query}'
    
    search_query = f'repo:{repo_name}+is:issue+is:closed+{labels_query}'
    
    url = f'https://api.github.com/search/issues?q={search_query}&sort=updated&order=desc&per_page=100'
    
    issues_scraped = 0
    
    while url and issues_scraped < MAX_ISSUES_PER_REPO:
        try:
            print(f"Fetching page: {url.split('&page=')[-1] if '&page=' in url else '1'}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            issues = data.get('items', [])
            
            if not issues:
                print("No more issues found for this query.")
                break

            for issue in issues:
                if issues_scraped >= MAX_ISSUES_PER_REPO:
                    break
                
                print(f"  - Processing Issue #{issue['number']}: {issue['title'][:50]}...")
                
                comments = get_issue_comments(issue['comments_url'], headers)
                
                record = {
                    'repo': repo_name,
                    'issue_number': issue['number'],
                    'issue_url': issue['html_url'],
                    'issue_title': issue['title'],
                    'issue_author': issue['user']['login'],
                    'issue_body': issue['body'],
                    'issue_labels': [label['name'] for label in issue['labels']],
                    'comments': comments
                }
                
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + '\n')
                
                issues_scraped += 1
                time.sleep(1)

            if 'next' in response.links:
                url = response.links['next']['url']
            else:
                url = None
            
            respectful_wait(response.headers)

        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] An API error occurred: {e}", file=sys.stderr)
            print("Waiting for 60 seconds before retrying...")
            time.sleep(60)
            
    print(f"--- Finished scraping for {repo_name}. Total issues saved: {issues_scraped} ---")

def get_existing_issue_counts():
    """Reads the output file to count issues already scraped for each repo."""
    counts = {}
    if not os.path.exists(OUTPUT_FILE):
        return counts
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                repo = record.get('repo')
                if repo:
                    counts[repo] = counts.get(repo, 0) + 1
            except json.JSONDecodeError:
                print(f"Warning: Could not decode a line in {OUTPUT_FILE}", file=sys.stderr)
    return counts

def main():
    """Main function to orchestrate the scraping process."""
    if not GITHUB_TOKEN:
        print("[FATAL ERROR] GITHUB_TOKEN not found in .env file or environment variables.", file=sys.stderr)
        print("Please create a .env file and add your token, then run the script again.", file=sys.stderr)
        sys.exit(1)
        
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # **NEW: Check for existing work before starting**
    existing_counts = get_existing_issue_counts()
    print("Found existing issue counts:", existing_counts)
    
    for repo, config in TARGET_CONFIG.items():
        if existing_counts.get(repo, 0) >= MAX_ISSUES_PER_REPO:
            print(f"\n--- Skipping {repo}: Already have {existing_counts.get(repo, 0)} issues (target is {MAX_ISSUES_PER_REPO}). ---")
            continue
        
        scrape_repo(repo, config['labels'], headers)
        
    print(f"\nAll scraping complete. Data saved to '{OUTPUT_FILE}'.")

if __name__ == '__main__':
    main()

