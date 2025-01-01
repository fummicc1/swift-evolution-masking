from datetime import datetime
import frontmatter
import glob
import os
import requests
import random
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.sane_lists import SaneListExtension
from markdown.extensions.nl2br import Nl2BrExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.footnotes import FootnoteExtension
from markdown.extensions.tables import TableExtension


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


def should_mask_word(word, inside_code_block, processes_metadata):
    if inside_code_block or word.startswith("```") or processes_metadata:
        return False
    if len(word) <= 2 or not any(c.isalnum() for c in word):
        return False
    return random.random() < 0.3


def mask_content(content):
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

        words = line.split()
        masked_words = [
            r"\_" * len(word) if should_mask_word(word, inside_code_block, processes_metadata) else word
            for word in words
        ]
        masked_line = " ".join(masked_words)

        if line.endswith((" ", "\t")):
            masked_line += line[len(line.rstrip()) :]

        masked_lines.append(masked_line)

    return "\n".join(masked_lines), metadatas


def upload_to_microcms(proposal_data):
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

    response = requests.post(endpoint, headers=headers, json=microcms_data)
    response.raise_for_status()
    return response.json()


def main():
    random.seed(42)
    proposal_files = sorted(list(glob.glob("proposals/*.md")))

    for file_path in proposal_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                post = frontmatter.loads(f.read())

            proposal_id = os.path.basename(file_path).split("-")[0]
            masked_content, metadatas = mask_content(post.content)

            proposal_data = {
                "title": metadatas["Title"],
                "content": masked_content,
                "proposal_id": proposal_id,
                "status": metadatas["Status"],
                "authors": metadatas["Authors"] or metadatas["Author"],
                "review_manager": metadatas["Review Manager"],
            }

            result = upload_to_microcms(proposal_data)
            print(f"Successfully uploaded proposal {proposal_id}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    main()
