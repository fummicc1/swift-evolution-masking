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
import en_core_web_md
from word_freq_hist import get_histogram_of_words, visualize_histogram_and_return_df, build_word_similarity_map
import boto3
from bs4 import BeautifulSoup
from typing import Dict, List, Set, Tuple, Any

# Load environment variables from .env file if it exists
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

nlp = en_core_web_md.load()


class Answer:
    """Class representing a masked word answer with options."""
    
    proposalId: str
    index: int
    answer: str
    options: list[str]

    def __init__(self, proposalId: str, index: int, answer: str, options: list[str] = None):
        """
        Initialize an Answer object.
        
        Args:
            proposalId: ID of the proposal
            index: Index of the answer in the document
            answer: The actual word that was masked
            options: List of similar words to serve as options
        """
        self.proposalId = proposalId
        self.index = index
        self.answer = answer
        self.options = options or []


def check_if_word_is_name(word: str) -> bool:
    """
    Check if a word is a noun using spaCy.
    
    Args:
        word: Word to check
        
    Returns:
        True if the word is a noun, False otherwise
    """
    global nlp
    doc = nlp(word)
    return any(ent.pos_ == "NOUN" for ent in doc)


def convert_markdown_to_html(markdown_content: str) -> str:
    """
    Convert markdown content to HTML.
    
    Args:
        markdown_content: Markdown content to convert
        
    Returns:
        HTML content
    """
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
    word: str, 
    inside_code_block: bool, 
    inside_inline_code: bool, 
    inside_hyperlink: bool, 
    processes_metadata: bool
) -> bool:
    """
    Determine if a word should be masked.
    
    Args:
        word: Word to check
        inside_code_block: Whether the word is inside a code block
        inside_inline_code: Whether the word is inside inline code
        inside_hyperlink: Whether the word is inside a hyperlink
        processes_metadata: Whether we're processing metadata
        
    Returns:
        True if the word should be masked, False otherwise
    """
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


class MarkdownParser:
    """Class to parse and mask markdown content."""
    
    def __init__(self, content: str, proposal_id: str, similarity_map: Dict[str, List[str]] = None):
        """
        Initialize the parser.
        
        Args:
            content: Markdown content to parse
            proposal_id: ID of the proposal
            similarity_map: Dictionary mapping words to similar words
        """
        self.content = content
        self.proposal_id = proposal_id
        self.similarity_map = similarity_map
        self.metadatas = {
            "Title": "",
            "Status": "",
            "Authors": "",
            "Author": "",
            "Review Manager": "",
            "Answer": [],
        }
        self.masked_words = []  # Words that will be masked
        self.lines = content.split("\n")
        
    def _update_metadata(self, line: str) -> None:
        """
        Update metadata from a line.
        
        Args:
            line: Line to parse for metadata
        """
        if line.startswith("# "):
            self.metadatas["Title"] = line[2:].strip()
            return
            
        if any(line[2:].startswith(key) for key in self.metadatas.keys()):
            key = line[2:].split(":")[0]
            prefix = len(key) + 2 + 1
            if prefix < len(line):
                self.metadatas[key] = line[prefix:].strip()
    
    def _check_markup_state(self, word: str, state: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Update and check the state of markdown markup elements.
        
        Args:
            word: Current word
            state: Current state dictionary
            
        Returns:
            Tuple of (is_inside_hyperlink, is_inside_inline_code)
        """
        # Reset hyperlink state if complete
        if all(state["inside_hyperlink"]):
            state["inside_hyperlink"] = [False, False, False, False]
            
        # Reset inline code state if complete
        if all(state["inside_inline_code"]):
            state["inside_inline_code"] = [False, False]
            
        # Update inline code state
        if "`" in word:
            if state["inside_inline_code"][0]:
                state["inside_inline_code"][1] = True
            else:
                state["inside_inline_code"][0] = True
                
        # Update hyperlink state
        if "[" in word:
            state["inside_hyperlink"][0] = True
        if state["inside_hyperlink"][0] and "]" in word:
            state["inside_hyperlink"][1] = True
        if state["inside_hyperlink"][1] and "(" in word:
            state["inside_hyperlink"][2] = True
        if state["inside_hyperlink"][2] and ")" in word:
            state["inside_hyperlink"][3] = True
            
        return any(state["inside_hyperlink"]), any(state["inside_inline_code"])
    
    def collect_words_to_mask(self) -> None:
        """Collect all words that will be masked in the first pass."""
        inside_code_block = False
        processes_metadata = True
        
        for line in self.lines:
            if line.startswith("##"):
                processes_metadata = False
                
            if line.startswith("#") or line.startswith("---") or not line.strip():
                continue
                
            if line.startswith("```"):
                inside_code_block = not inside_code_block
                
            if inside_code_block:
                continue
                
            words = line.split()
            state = {
                "inside_inline_code": [False, False],
                "inside_hyperlink": [False, False, False, False]
            }
            
            for word in words:
                is_inside_hyperlink, is_inside_inline_code = self._check_markup_state(word, state)
                
                contains_punctuation = self.contains_punctuation(word)
                
                if should_mask_word(
                    word=word,
                    inside_code_block=inside_code_block,
                    inside_inline_code=is_inside_inline_code,
                    inside_hyperlink=is_inside_hyperlink,
                    processes_metadata=processes_metadata,
                ):
                    masked_word = word[:-1].lower() if contains_punctuation else word.lower()
                    self.masked_words.append(masked_word)
    
    def contains_punctuation(self, word: str) -> bool:
        """
        Check if a word is a punctuation.
        """
        return word[-1] in [
            ".", ",", "!", "?", ":", ";", "-", "_", "~", "|", "=", "+", "*", "/", "\\", "@"
        ]
    
    def mask_word(self, word: str) -> Tuple[str, bool]:
        """
        Mask a word by replacing it with squares.
        
        Args:
            word: Word to mask
            
        Returns:
            Tuple of (masked_word, was_masked)
        """
        contains_punctuation = self.contains_punctuation(word)
        blank = r"＿" * 5
        return blank, contains_punctuation
    
    def get_similar_word_options(self, word: str) -> List[str]:
        """
        Get similar words as options for a masked word.
        
        Args:
            word: Word to find options for
            
        Returns:
            List of similar words
        """
        options = []
        word = word[:-1].lower() if self.contains_punctuation(word) else word.lower()
        if self.similarity_map and word in self.similarity_map:
            similar_words = self.similarity_map.get(word, [])
            if similar_words:
                # 最も類似度の高い上位3つの単語を選択（ランダムサンプリングしない）
                options = similar_words[:min(3, len(similar_words))]
        return options
    
    def mask_content(self) -> str:
        """
        Mask the content and collect answers.
        
        Returns:
            Masked content as a string
        """
        # First collect all words to mask
        self.collect_words_to_mask()
        
        # Second pass to actually mask the content
        masked_lines = []
        inside_code_block = False
        processes_metadata = True
        
        for line in self.lines:
            if line.startswith("# "):
                self.metadatas["Title"] = line[2:].strip()
                masked_lines.append(line)
                continue
                
            if line.startswith("##"):
                processes_metadata = False
                
            if line.startswith("#") or line.startswith("---") or not line.strip():
                masked_lines.append(line)
                continue
                
            if processes_metadata:
                self._update_metadata(line)
                if any(line[2:].startswith(key) for key in self.metadatas.keys()):
                    masked_lines.append(line)
                    continue
                    
            if line.startswith("```"):
                inside_code_block = not inside_code_block
                
            if inside_code_block:
                masked_lines.append(line)
                continue
                
            words = line.split()
            state = {
                "inside_inline_code": [False, False],
                "inside_hyperlink": [False, False, False, False]
            }
            masked_words = []
            
            for word in words:
                is_inside_hyperlink, is_inside_inline_code = self._check_markup_state(word, state)
                
                if should_mask_word(
                    word=word,
                    inside_code_block=inside_code_block,
                    inside_inline_code=is_inside_inline_code,
                    inside_hyperlink=is_inside_hyperlink,
                    processes_metadata=processes_metadata,
                ):
                    masked_word, contains_punctuation = self.mask_word(word)
                    if contains_punctuation:
                        # masked_word is a blank, so we need to add the punctuation back.
                        masked_words.append(masked_word + word[-1])
                    else:
                        masked_words.append(masked_word)
                    
                    # Get options for this word
                    options = self.get_similar_word_options(word)
                    
                    # Add to answers
                    answer_word = word if not contains_punctuation else word[:-1]
                    self.metadatas["Answer"].append(
                        Answer(
                            proposalId=self.proposal_id,
                            index=len(self.metadatas["Answer"]),
                            answer=answer_word,
                            options=options
                        )
                    )
                else:
                    masked_words.append(word)
                    
            masked_line = " ".join(masked_words)
            
            # Preserve trailing whitespace
            if line.endswith((" ", "\t")):
                masked_line += line[len(line.rstrip()):]
                
            masked_lines.append(masked_line)
            
        return "\n".join(masked_lines)
    
    def get_masked_content_and_metadata(self) -> Tuple[str, Dict]:
        """
        Get the masked content and metadata.
        
        Returns:
            Tuple of (masked_content, metadata)
        """
        masked_content = self.mask_content()
        return masked_content, self.metadatas


def simplify_html_structure(html_content: str) -> str:
    """
    Simplify complex HTML structure for better compatibility with microCMS.
    
    Args:
        html_content: HTML content to simplify
        
    Returns:
        Simplified HTML content
    """
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


class MicroCMSManager:
    """Class for managing interactions with microCMS."""
    
    def __init__(self):
        """Initialize with API keys from environment variables."""
        self.api_key = os.environ["MICROCMS_API_KEY"]
        self.domain = os.environ["MICROCMS_SERVICE_DOMAIN"]
        self.endpoint = f"https://{self.domain}.microcms.io/api/v1/proposals"
        self.headers = {"X-MICROCMS-API-KEY": self.api_key, "Content-Type": "application/json"}
        self.all_proposals = []
        
    def fetch_all_proposals(self):
        """Fetch all proposals from microCMS."""
        print("Fetching all proposals from microcms...")
        # First, get all proposals. Iterate over all pages (around 500 contents in total).
        offset = 0
        for _ in range(10):
            response = requests.get(
                f"{self.endpoint}?limit=100&offset={offset}", headers=self.headers
            )
            response.raise_for_status()
            proposals = response.json()["contents"]
            self.all_proposals.extend(proposals)
            offset += 100
        print(f"Done fetching all proposals from microcms ({len(self.all_proposals)} proposals)")
        
    def delete_proposal(self, proposal_id: str):
        """
        Delete a proposal from microCMS.
        
        Args:
            proposal_id: ID of the proposal to delete
        """
        print(f"Deleting proposal {proposal_id} from microcms...")
        try:
            contents = list(
                filter(
                    lambda proposal: proposal["proposalId"] == proposal_id, self.all_proposals
                )
            )
            for content in contents:
                content_id = content["id"]
                delete_endpoint = f"{self.endpoint}/{content_id}"
                response = requests.delete(delete_endpoint, headers=self.headers)
                response.raise_for_status()
            print(f"Successfully deleted proposal {proposal_id} from microcms")
        except Exception as e:
            # Not raise an error in case the proposal does not exist.
            print(f"Error deleting proposal {proposal_id} from microcms: {str(e)}")
            
    def upload_proposal(self, proposal_data: dict):
        """
        Upload a proposal to microCMS with retry logic.
        
        Args:
            proposal_data: Dictionary containing proposal data
            
        Returns:
            Response JSON from microCMS
        """
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                print(f"Sending request to microCMS for proposal {proposal_data['proposal_id']} (attempt {attempt + 1}/{max_retries})...")
                
                html_content = convert_markdown_to_html(proposal_data["content"])

                microcms_data = {
                    "title": proposal_data["title"],
                    "content": html_content,
                    "proposalId": proposal_data["proposal_id"],
                    "status": proposal_data["status"],
                    "authors": proposal_data["authors"],
                    "reviewManager": proposal_data.get("review_manager", ""),
                }

                response = requests.post(self.endpoint, headers=self.headers, json=microcms_data)
                
                if response.status_code == 502:
                    print("Retrying with simplified content due to 502 error")
                    simplified_html = simplify_html_structure(html_content)
                    microcms_data["content"] = simplified_html
                    response = requests.post(self.endpoint, headers=self.headers, json=microcms_data)
                
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


class R2Manager:
    """Class for managing interactions with Cloudflare R2."""
    
    def __init__(self):
        """Initialize with API keys from environment variables."""
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        )
        self.bucket_name = os.environ["R2_BUCKET_NAME"]
        
    def upload_answers(self, answers_data: List[Answer]):
        """
        Upload answers data to R2.
        
        Args:
            answers_data: List of Answer objects
        """
        # Convert Answer objects to a dictionary with proposalId as keys
        answers_json = {}
        for answer in answers_data:
            if answer.proposalId not in answers_json:
                answers_json[answer.proposalId] = []

            answers_json[answer.proposalId].append(
                {
                    "index": answer.index, 
                    "answer": answer.answer,
                    "options": answer.options
                }
            )
            
        # Upload to R2
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key="answers.json",
                Body=json.dumps(answers_json, ensure_ascii=False, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            print(f"Successfully uploaded answers to R2: answers.json")
        except Exception as e:
            print(f"Error uploading answers to R2: {str(e)}")
            
    def upload_word_freq_hist(self, word_freq_hist_df: pd.DataFrame):
        """
        Upload word frequency histogram to R2.
        
        Args:
            word_freq_hist_df: DataFrame containing word frequency histogram
        """
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key="word_freq_hist.json",
                Body=word_freq_hist_df.to_json(orient="records").encode("utf-8"),
                ContentType="application/json",
            )
            print(f"Successfully uploaded word frequency histogram to R2: word_freq_hist.json")
        except Exception as e:
            print(f"Error uploading word frequency histogram to R2: {str(e)}")
            
    def upload_similarity_map(self, similarity_map: Dict[str, List[str]]):
        """
        Upload similarity map to R2.
        
        Args:
            similarity_map: Dictionary mapping words to similar words
        """
        if not similarity_map:
            return
            
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key="similarity_map.json",
                Body=json.dumps(similarity_map, ensure_ascii=False, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            print(f"Successfully uploaded similarity map to R2: similarity_map.json")
        except Exception as e:
            print(f"Error uploading similarity map to R2: {str(e)}")
            
    def upload_all(self, answers_data: List[Answer], word_freq_hist_df: pd.DataFrame, similarity_map: Dict[str, List[str]] = None):
        """
        Upload all data to R2.
        
        Args:
            answers_data: List of Answer objects
            word_freq_hist_df: DataFrame containing word frequency histogram
            similarity_map: Dictionary mapping words to similar words
        """
        self.upload_answers(answers_data)
        self.upload_word_freq_hist(word_freq_hist_df)
        
        if similarity_map:
            self.upload_similarity_map(similarity_map)


def collect_all_nouns(proposal_files: List[str]) -> Tuple[Set[str], Counter]:
    """
    Collect all nouns from all proposal files.
    
    Args:
        proposal_files: List of proposal file paths
        
    Returns:
        Tuple of (set of unique nouns, Counter of word frequencies)
    """
    print("Collecting all nouns from all documents...")
    all_word_freq_hists = Counter()
    all_nouns = set()
    
    for file_path in proposal_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                post = frontmatter.loads(f.read())
            
            # Extract all nouns from the document
            word_freq_hist = get_histogram_of_words(nlp, post.content)
            all_word_freq_hists += word_freq_hist
            # Add all nouns to the set
            all_nouns.update(word_freq_hist.keys())
            
        except Exception as e:
            print(f"Error collecting nouns from {file_path}: {str(e)}")
            continue
    
    print(f"Collected {len(all_nouns)} unique nouns from all documents")
    return all_nouns, all_word_freq_hists


def process_proposal_file(
    file_path: str, 
    microcms_manager: MicroCMSManager, 
    similarity_map: Dict[str, List[str]]
) -> List[Answer]:
    """
    Process a single proposal file.
    
    Args:
        file_path: Path to the proposal file
        microcms_manager: MicroCMSManager instance
        similarity_map: Dictionary mapping words to similar words
        
    Returns:
        List of Answer objects
    """
    answers = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.loads(f.read())

        proposal_id = os.path.basename(file_path).split("-")[0]
        print(f"Processing proposal {proposal_id}...")
        
        # Use the MarkdownParser to mask content
        parser = MarkdownParser(post.content, proposal_id, similarity_map)
        masked_content, metadatas = parser.get_masked_content_and_metadata()
        answers.extend(metadatas["Answer"])

        proposal_data = {
            "title": metadatas["Title"],
            "content": masked_content,
            "proposal_id": proposal_id,
            "status": metadatas["Status"],
            "authors": metadatas["Authors"] or metadatas["Author"],
            "review_manager": metadatas["Review Manager"],
        }
        
        # Before uploading, delete the proposal from microcms if it exists.
        microcms_manager.delete_proposal(proposal_id)
        print(f"\nUploading proposal {proposal_id} to microcms...")
        result = microcms_manager.upload_proposal(proposal_data)
        print(f"Successfully uploaded proposal {proposal_id}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        
    return answers


def main():
    try:
        random.seed(42)
        print("Starting preprocessing of microCMS data...")
        
        # Initialize managers
        microcms_manager = MicroCMSManager()
        r2_manager = R2Manager()
        
        # Fetch all existing proposals
        microcms_manager.fetch_all_proposals()
        
        # Get all proposal files
        proposal_files = sorted(list(glob.glob("proposals/*.md")))
        print(f"Found {len(proposal_files)} proposal files to process")

        # First pass: collect all nouns and build similarity map
        all_nouns, all_word_freq_hists = collect_all_nouns(proposal_files)
        
        # Build the similarity map for all collected nouns
        print("Building similarity map for all nouns...")
        similarity_map = build_word_similarity_map(nlp, all_nouns)
        print(f"Finished building similarity map for {len(similarity_map)} words")

        # Second pass: mask content and process documents
        print("Processing all documents...")
        all_answers = []
        
        for file_path in proposal_files:
            answers = process_proposal_file(file_path, microcms_manager, similarity_map)
            all_answers.extend(answers)

        print("Generating word frequency histogram...")
        word_freq_hist_df = visualize_histogram_and_return_df(
            all_word_freq_hists, write_to_file=True
        )

        print("Uploading data to R2...")
        # Upload all data to R2
        r2_manager.upload_all(all_answers, word_freq_hist_df, similarity_map)
        
        print("Process completed successfully")

    except Exception as e:
        print(f"Fatal error in main process: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise  # Re-raise the exception to ensure the GitHub Action fails


def upload_to_r2(answers_data, word_freq_hist_df: pd.DataFrame, similarity_map=None):
    """
    Legacy function that uses the new R2Manager class.
    
    Args:
        answers_data: List of Answer objects
        word_freq_hist_df: DataFrame containing word frequency histogram
        similarity_map: Dictionary mapping words to similar words
    """
    manager = R2Manager()
    manager.upload_all(answers_data, word_freq_hist_df, similarity_map)


def preprocess_microcms_data():
    """Legacy function that uses the new MicroCMSManager class."""
    manager = MicroCMSManager()
    manager.fetch_all_proposals()
    global all_proposals
    all_proposals = manager.all_proposals


def delete_proposal(proposal_id: str):
    """
    Legacy function that uses the new MicroCMSManager class.
    
    Args:
        proposal_id: ID of the proposal to delete
    """
    manager = MicroCMSManager()
    manager.all_proposals = all_proposals
    manager.delete_proposal(proposal_id)


def upload_to_microcms(proposal_data):
    """
    Legacy function that uses the new MicroCMSManager class.
    
    Args:
        proposal_data: Dictionary containing proposal data
        
    Returns:
        Response JSON from microCMS
    """
    manager = MicroCMSManager()
    return manager.upload_proposal(proposal_data)


if __name__ == "__main__":
    main()
