from datetime import datetime
import frontmatter
import glob
import os
import requests
import random


def should_mask_word(word):
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
        "Authors": "",
        "Review Manager": "",
    }

    for line in lines:
        if line.startswith("# "):
            processes_metadata = True
            metadatas["Title"] = line[2:].strip()
            masked_lines.append(line)
            continue

        if line.startswith("#") or line.startswith("---") or not line.strip():
            masked_lines.append(line)
            continue

        if processes_metadata:
            if any(line[2:].startswith(key) for key in metadatas.keys()):
                key = line[2:].split(":")[0]
                metadatas[key] = line[len(key) + 2 :].strip()
                masked_lines.append(line)
                continue
            else:
                processes_metadata = False

        words = line.split()
        masked_words = [
            "_" * len(word) if should_mask_word(word) else word for word in words
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

    microcms_data = {
        "title": proposal_data["title"],
        "content": proposal_data["content"],
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
    proposal_files = glob.glob("proposals/*.md")

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
                "authors": metadatas["Authors"],
                "review_manager": metadatas["Review Manager"],
            }

            result = upload_to_microcms(proposal_data)
            print(f"Successfully uploaded proposal {proposal_id}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    main()
