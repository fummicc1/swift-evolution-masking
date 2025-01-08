from collections import Counter
from datetime import datetime
import json
import time
import frontmatter
import glob
import os
import requests
import random
import markdown
import pandas as pd
import traceback
from dotenv import load_dotenv
from pathlib import Path
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.sane_lists import SaneListExtension
from markdown.extensions.nl2br import Nl2BrExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.footnotes import FootnoteExtension
from markdown.extensions.tables import TableExtension
import en_core_web_sm
from word_freq_hist import get_histogram_of_words, visualize_histogram_and_return_df
import boto3
from bs4 import BeautifulSoup

# Load environment variables from .env file if it exists
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

nlp = en_core_web_sm.load()


class Answer:
    proposalId: str
    index: int
    answer: str

    def __init__(self, proposalId: str, index: int, answer: str):
        self.proposalId = proposalId
        self.index = index
        self.answer = answer


def check_if_word_is_name(word):
    global nlp
    doc = nlp(word)
    return any(ent.pos_ == "NOUN" for ent in doc)


def convert_markdown_to_html(markdown_content):
    extensions = [
        CodeHiliteExtension(),
        SaneListExtension(),
        Nl2BrExtension(),
        FencedCodeExtension(),
        FootnoteExtension(),
        TableExtension(),
    ]
    html_content = markdown.markdown(markdown_content, extensions=extensions)
    return html_content


def should_mask_word(
    word, inside_code_block, inside_inline_code, inside_hyperlink, processes_metadata
):
    if (
        inside_code_block
        or inside_inline_code
        or inside_hyperlink
        or word.startswith("```")
        or processes_metadata
    ):
        return False
    if len(word) <= 2 or not any(c.isalnum() for c in word):
        return False
    if not check_if_word_is_name(word):
        return False
    return random.random() < 0.3


def mask_content(content: str, proposal_id: str):
    lines = content.split("\n")
    masked_lines = []
    processes_metadata = True
    metadatas = {
        "Title": "",
        "Status": "",
        # Format is inconsistent in the original data
        "Authors": "",
        "Author": "",
        "Review Manager": "",
        "Answer": [],
    }
    inside_code_block = False

    for line in lines:
        if line.startswith("# "):
            metadatas["Title"] = line[2:].strip()
            masked_lines.append(line)
            continue

        if line.startswith("##"):
            processes_metadata = False

        if line.startswith("#") or line.startswith("---") or not line.strip():
            masked_lines.append(line)
            continue

        if processes_metadata:
            if any(line[2:].startswith(key) for key in metadatas.keys()):
                key = line[2:].split(":")[0]
                prefix = len(key) + 2 + 1
                if prefix < len(line):
                    metadatas[key] = line[prefix:].strip()
                masked_lines.append(line)
                continue

        if line.startswith("```"):
            inside_code_block = not inside_code_block

        if inside_code_block:
            masked_lines.append(line)
            continue

        words: list[str] = line.split()
        inside_inline_code = [False, False]
        inside_hyperlink = [False, False, False, False]
        masked_words: list[str] = []

        for word in words:

            if all(inside_hyperlink):
                inside_hyperlink = [False, False, False, False]

            if all(inside_inline_code):
                inside_inline_code = [False, False]

            if "`" in word:
                # If inline code is not closed, we should not mask the word.
                if inside_inline_code[0]:
                    inside_inline_code[1] = True
                else:
                    inside_inline_code[0] = True

            if "[" in word:
                inside_hyperlink[0] = True
            if inside_hyperlink[0] and "]" in word:
                inside_hyperlink[1] = True
            if inside_hyperlink[1] and "(" in word:
                inside_hyperlink[2] = True
            if inside_hyperlink[2] and ")" in word:
                inside_hyperlink[3] = True

            is_inside_hyperlink = any(inside_hyperlink)
            is_inside_inline_code = any(inside_inline_code)

            if should_mask_word(
                word=word,
                inside_code_block=inside_code_block,
                inside_inline_code=is_inside_inline_code,
                inside_hyperlink=is_inside_hyperlink,
                processes_metadata=processes_metadata,
            ):
                contains_punctuation = word[-1] in [
                    ".",
                    ",",
                    "!",
                    "?",
                    ":",
                    ";",
                    "-",
                    "_",
                    "~",
                    "|",
                    "=",
                    "+",
                    "*",
                    "/",
                    "\\",
                    "@",
                ]
                if contains_punctuation:
                    masked_word_and_punctuation = r"◻︎" * (len(word) - 1) + word[-1]
                    masked_words.append(masked_word_and_punctuation)
                else:
                    masked_words.append(r"◻︎" * len(word))
                metadatas["Answer"].append(
                    Answer(
                        proposalId=proposal_id,
                        index=len(metadatas["Answer"]),
                        answer=word,
                    )
                )
            else:
                masked_words.append(word)

        masked_line = " ".join(masked_words)

        if line.endswith((" ", "\t")):
            masked_line += line[len(line.rstrip()) :]

        masked_lines.append(masked_line)

    return "\n".join(masked_lines), metadatas

def simplify_html_structure(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. Split and simplify deeply nested lists
    def flatten_deep_lists(ul_element, current_depth=0, max_depth=2):
        if current_depth > max_depth:
            new_section = soup.new_tag('div', attrs={'class': 'section'})
            new_list = soup.new_tag('ul')
            
            for li in ul_element.find_all('li', recursive=False):
                new_li = soup.new_tag('li')
                new_li.string = li.get_text()
                new_list.append(new_li)
            
            new_section.append(new_list)
            ul_element.replace_with(new_section)
        else:
            for nested_ul in ul_element.find_all('ul', recursive=False):
                flatten_deep_lists(nested_ul, current_depth + 1)
    
    # 2. Split long lists into sections
    def split_long_lists(ul_element, max_items=10):
        items = ul_element.find_all('li', recursive=False)
        if len(items) > max_items:
            sections = []
            for i in range(0, len(items), max_items):
                new_section = soup.new_tag('div', attrs={'class': 'list-section'})
                new_list = soup.new_tag('ul')
                
                for item in items[i:i + max_items]:
                    new_list.append(item.copy())
                
                new_section.append(new_list)
                sections.append(new_section)
            
            parent = ul_element.parent
            ul_element.replace_with(*sections)
    
    # 3. Separate complex content from lists
    def extract_complex_content(ul_element):
        for li in ul_element.find_all('li'):
            # Extract code blocks and tables
            for element in li.find_all(['pre', 'code', 'table']):
                new_div = soup.new_tag('div', attrs={'class': f'{element.name}-section'})
                new_div.append(element.extract())
                li.insert_after(new_div)
    
    # Main processing
    for ul in soup.find_all('ul'):
        flatten_deep_lists(ul)
        split_long_lists(ul)
        extract_complex_content(ul)
    
    return str(soup)


def upload_to_microcms(proposal_data):
    """Upload proposal data to microCMS with retry logic"""
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            api_key = os.environ["MICROCMS_API_KEY"]
            domain = os.environ["MICROCMS_SERVICE_DOMAIN"]
            endpoint = f"https://{domain}.microcms.io/api/v1/proposals"

            headers = {"X-MICROCMS-API-KEY": api_key, "Content-Type": "application/json"}

            html_content = convert_markdown_to_html(proposal_data["content"])

            microcms_data = {
                "title": proposal_data["title"],
                "content": html_content,
                "proposalId": proposal_data["proposal_id"],
                "status": proposal_data["status"],
                "authors": proposal_data["authors"],
                "reviewManager": proposal_data.get("review_manager", ""),
            }

            print(f"Sending request to microCMS for proposal {proposal_data['proposal_id']} (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(endpoint, headers=headers, json=microcms_data)
            
            if response.status_code == 502:
                print("Retrying with simplified content due to 502 error")
                simplified_html = simplify_html_structure(html_content)
                microcms_data["content"] = simplified_html
                response = requests.post(endpoint, headers=headers, json=microcms_data)
            
            if response.status_code >= 400:
                print(f"Error response from microCMS: {response.status_code}")
                print(f"Response content: {response.text}")
                response.raise_for_status()
                
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Network error while uploading to microCMS (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"Response content: {e.response.text}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise
        except Exception as e:
            print(f"Unexpected error while uploading to microCMS: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())
            raise


all_proposals = []


def delete_proposal(proposal_id: str):
    print(f"Deleting proposal {proposal_id} from microcms...")
    api_key = os.environ["MICROCMS_API_KEY"]
    domain = os.environ["MICROCMS_SERVICE_DOMAIN"]
    endpoint = f"https://{domain}.microcms.io/api/v1/proposals"

    headers = {"X-MICROCMS-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        contents = list(
            filter(
                lambda proposal: proposal["proposalId"] == proposal_id, all_proposals
            )
        )
        for content in contents:
            content_id = content["id"]
            delete_endpoint = f"{endpoint}/{content_id}"
            response = requests.delete(delete_endpoint, headers=headers)
            response.raise_for_status()
        print(f"Successfully deleted proposal {proposal_id} from microcms")
    except Exception as e:
        # Not raise an error in case the proposal does not exist.
        print(f"Error deleting proposal {proposal_id} from microcms: {str(e)}")


def preprocess_microcms_data():
    print("Fetching all proposals from microcms...")
    # Delete all proposals in microcms
    api_key = os.environ["MICROCMS_API_KEY"]
    domain = os.environ["MICROCMS_SERVICE_DOMAIN"]
    endpoint = f"https://{domain}.microcms.io/api/v1/proposals"

    headers = {"X-MICROCMS-API-KEY": api_key, "Content-Type": "application/json"}
    # First, get all proposals. Iterate over all pages (around 500 contents in total).
    offset = 0
    for _ in range(10):
        response = requests.get(
            f"{endpoint}?limit=100&offset={offset}", headers=headers
        )
        response.raise_for_status()
        proposals = response.json()["contents"]
        all_proposals.extend(proposals)
        offset += 100
    print("Done fetching all proposals from microcms")


def upload_to_r2(answers_data, word_freq_hist_df: pd.DataFrame):
    """Upload answers data to Cloudflare R2"""
    r2 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )

    # Convert Answer objects to a dictionary with proposalId as keys
    answers_json = {}
    for answer in answers_data:
        if answer.proposalId not in answers_json:
            answers_json[answer.proposalId] = []

        answers_json[answer.proposalId].append(
            {"index": answer.index, "answer": answer.answer}
        )
    # Upload to R2
    try:
        r2.put_object(
            Bucket=os.environ["R2_BUCKET_NAME"],
            Key="answers.json",
            Body=json.dumps(answers_json, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"Successfully uploaded answers to R2: answers.json")
    except Exception as e:
        print(f"Error uploading answers to R2: {str(e)}")

    # Upload word frequency histogram to R2
    try:
        r2.put_object(
            Bucket=os.environ["R2_BUCKET_NAME"],
            Key="word_freq_hist.json",
            Body=word_freq_hist_df.to_json(orient="records").encode("utf-8"),
            ContentType="application/json",
        )
        print(
            f"Successfully uploaded word frequency histogram to R2: word_freq_hist.json"
        )
    except Exception as e:
        print(f"Error uploading word frequency histogram to R2: {str(e)}")


def main():
    try:
        random.seed(42)
        print("Starting preprocessing of microCMS data...")
        preprocess_microcms_data()
        proposal_files = sorted(list(glob.glob("proposals/*.md")))
        print(f"Found {len(proposal_files)} proposal files to process")

        all_word_freq_hists = Counter()
        all_answers = []

        for file_path in proposal_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    post = frontmatter.loads(f.read())

                proposal_id = os.path.basename(file_path).split("-")[0]
                print(f"Processing proposal {proposal_id}...")
                
                masked_content, metadatas = mask_content(post.content, proposal_id)
                word_freq_hist = get_histogram_of_words(nlp, post.content)
                all_word_freq_hists += word_freq_hist
                all_answers.extend(metadatas["Answer"])

                proposal_data = {
                    "title": metadatas["Title"],
                    "content": masked_content,
                    "proposal_id": proposal_id,
                    "status": metadatas["Status"],
                    "authors": metadatas["Authors"] or metadatas["Author"],
                    "review_manager": metadatas["Review Manager"],
                }
                # Before uploading, delete the proposal from microcms if it exists.
                delete_proposal(proposal_id)
                print(f"\nUploading proposal {proposal_id} to microcms...")
                result = upload_to_microcms(proposal_data)
                print(f"Successfully uploaded proposal {proposal_id}")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                continue

        print("Generating word frequency histogram...")
        word_freq_hist_df = visualize_histogram_and_return_df(
            all_word_freq_hists, write_to_file=True
        )

        print("Uploading data to R2...")
        # Upload all answers to R2
        upload_to_r2(all_answers, word_freq_hist_df)
        
        print("Process completed successfully")

    except Exception as e:
        print(f"Fatal error in main process: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise  # Re-raise the exception to ensure the GitHub Action fails


if __name__ == "__main__":
    main()
