# Builds a supervised fine-tuning dataset from a PDF:
#   1. split the PDF into chunks (saved in a SQLite file, so a stopped run can resume)
#   2. ask the teacher model in Ollama for question-answer pairs about each chunk
#   3. export train / validation / test JSONL files for ollama_model.py
#
# Run it from anywhere:
#   python fine_tuning/sft_datagen.py --pdf fine_tuning/data/7110.65BB_Bsc_w_Chg_1_2_and_3_dtd_7-9-26_Final.pdf --limit 10
#
# Output (always inside the fine_tuning folder, no matter where you run it from):
#   fine_tuning/qa_database.sqlite                 progress and every generated pair
#   fine_tuning/hf_qa_dataset/train.jsonl          ready for TRL's SFTTrainer (prompt + completion only)
#   fine_tuning/hf_qa_dataset/validation.jsonl
#   fine_tuning/hf_qa_dataset/test.jsonl
#   fine_tuning/hf_qa_dataset/*_sources.jsonl      the same pairs plus page and evidence, for checking by hand

import argparse
import hashlib
import json
import random
import re
import sqlite3
from pathlib import Path

import pymupdf
from ollama import Client
from tqdm import tqdm

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Every file is kept next to this script, so it doesn't matter which folder you run it from.
HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "qa_database.sqlite"
OUTPUT_DIR = HERE / "hf_qa_dataset"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUESTION_MODEL = 'gemma4:12b'
SEED = 42

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT
);

CREATE TABLE IF NOT EXISTS qa (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    evidence TEXT NOT NULL,
    UNIQUE(node_id, question),
    FOREIGN KEY(node_id) REFERENCES nodes(node_id)
);
"""


# ---------------------------------------------------------------------------
# 1. PDF -> chunks
# ---------------------------------------------------------------------------


def normalize_text(text):
    text = re.sub(r"[ \t]+", " ", text)  # runs of spaces and tabs -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)  # three or more blank lines -> one blank line
    return text.strip()


def extract_pages(path):
    documents = []

    with pymupdf.open(path) as pdf:
        for idx, page in enumerate(pdf):
            text = normalize_text(page.get_text("text", sort=True))

            if len(text) < 50:
                continue

            documents.append(Document(text=text, metadata={"page": idx + 1, "source": str(path)}))

    return documents


def semantic_chunk(documents):
    embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    semantic_splitter = SemanticSplitterNodeParser(embed_model=embed, buffer_size=1, breakpoint_percentile_threshold=95)
    size_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    semantic_nodes = semantic_splitter.get_nodes_from_documents(documents, show_progress=True)

    final_nodes = []
    for node in semantic_nodes:
        pieces = size_splitter.get_nodes_from_documents([Document(text=node.text, metadata=node.metadata)])
        final_nodes.extend(pieces)

    return final_nodes


# ---------------------------------------------------------------------------
# 2. Database: remembers chunks and pairs, so a stopped run picks up where it left off
# ---------------------------------------------------------------------------


def init_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def save_nodes(conn, nodes):
    records = []

    for node in nodes:
        text = node.text.strip()

        if len(text) < 100:
            continue

        page = int(node.metadata["page"])
        node_id = hashlib.sha256(f"{page}:{text}".encode("utf-8")).hexdigest()
        records.append((node_id, page, text))

    with conn:
        conn.executemany("INSERT OR IGNORE INTO nodes (node_id, page, text) VALUES (?, ?, ?)", records)

    return len(records)


# ---------------------------------------------------------------------------
# 3. Teacher model -> question-answer pairs
# ---------------------------------------------------------------------------


def generate_qa(client, model, source, max_pairs=3):
    prompt = f"""You are generating supervised fine-tuning examples
                from a technical document.

                Generate up to {max_pairs} distinct question-answer pairs.

                Requirements:
                1. Questions must be answerable using the source.
                2. Answers must be factually supported by the source.
                3. Prefer technical and conceptual questions.
                4. Include reasoning questions when appropriate.
                5. Avoid repetitive or trivial questions.
                6. Do not invent facts.
                7. Copy an exact supporting quote into "evidence".
                8. If the source is uninformative, return no pairs.

                Return a JSON object:
                {{
                "pairs": [
                    {{
                    "question": "...",
                    "answer": "...",
                    "evidence": "exact quote from source"
                    }}
                ]
                }}

                SOURCE:
                {source}
            """

    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        think=False,
        options={'temperature': 0, 'num_ctx': 4096},
    )

    data = json.loads(response.message.content)
    pairs = data.get("pairs", [])

    if not isinstance(pairs, list):
        raise ValueError("Invalid pairs field")

    valid = []
    for pair in pairs[:max_pairs]:
        if not isinstance(pair, dict):
            continue

        q = str(pair.get("question", "")).strip()
        a = str(pair.get("answer", "")).strip()
        e = str(pair.get("evidence", "")).strip()

        if not (q and a and e): # all must exist for this to continue.
            continue

        # Only keep pairs whose quote really is in the source.
        if e not in source:
            continue

        valid.append((q, a, e))

    return valid


def generate_all(conn, model, max_pairs, limit):
    client = Client()

    pending = conn.execute("SELECT node_id, text FROM nodes WHERE status != 'done' ORDER BY page, node_id").fetchall()

    if limit:
        pending = pending[:limit]

    for node_id, source in tqdm(pending):
        for attempt in range(3):
            try:
                pairs = generate_qa(client, model, source, max_pairs)

                with conn:
                    conn.executemany(
                        "INSERT OR IGNORE INTO qa (node_id, question, answer, evidence) VALUES (?, ?, ?, ?)",
                        [(node_id, q, a, e) for q, a, e in pairs],
                    )
                    conn.execute("UPDATE nodes SET status='done', error=NULL WHERE node_id=?", (node_id,))
                break

            except Exception as exc:
                conn.execute("UPDATE nodes SET status='failed', error=? WHERE node_id=?", (str(exc), node_id))


# ---------------------------------------------------------------------------
# 4. Export: train / validation / test JSONL for ollama_model.py
# ---------------------------------------------------------------------------
def load_qa(conn):
    rows = conn.execute(
        """
        SELECT qa.question, qa.answer, qa.evidence, nodes.page, nodes.node_id
        FROM qa
        JOIN nodes ON qa.node_id = nodes.node_id
        WHERE nodes.status = 'done'
        ORDER BY nodes.page, qa.id
        """
    ).fetchall()

    examples = []
    seen = set()

    for question, answer, evidence, page, node_id in rows:
        # Skip questions that are the same apart from case and spacing.
        normalized = " ".join(question.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)

        examples.append({"question": question, "answer": answer, "evidence": evidence, "page": page, "node_id": node_id})

    return examples


def make_splits(examples):
    """Splits 80/10/10 by page, so the same page never shows up in both train and test."""
    page_groups = {}
    for ex in examples:
        page_groups.setdefault(ex["page"], []).append(ex)

    pages = list(page_groups.keys())
    random.Random(SEED).shuffle(pages)

    n = len(pages)
    if n < 3:
        raise ValueError(
            "Need pairs from at least 3 different pages to make train/validation/test splits, but only have "
            + str(n)
            + ". Run again with a bigger --limit."
        )

    n_train = int(n * 0.8)
    n_val = max(1, int(n * 0.1))

    train_pages = set(pages[:n_train])
    val_pages = set(pages[n_train : n_train + n_val])

    splits = {"train": [], "validation": [], "test": []}

    for ex in examples:
        if ex["page"] in train_pages:
            split = "train"
        elif ex["page"] in val_pages:
            split = "validation"
        else:
            split = "test"

        splits[split].append(ex)

    return splits


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_dataset(conn, output_dir=OUTPUT_DIR):
    examples = load_qa(conn)
    splits = make_splits(examples)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for split, split_examples in splits.items():
        # Conversational prompt-completion format. TRL's SFTTrainer reads this directly
        # and only calculates loss on the completion. Only these two fields are kept,
        # so the trainer doesn't trip over extra columns.
        training_records = [
            {
                "prompt": [{"role": "user", "content": ex["question"]}],
                "completion": [{"role": "assistant", "content": ex["answer"]}],
            }
            for ex in split_examples
        ]
        write_jsonl(output / f"{split}.jsonl", training_records)

        # The same pairs with where they came from, for checking by hand.
        write_jsonl(output / f"{split}_sources.jsonl", split_examples)

        print(f"Wrote {len(training_records)} pairs to {output / f'{split}.jsonl'}")

    return output


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--model", default=QUESTION_MODEL)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="only process this many chunks (0 = all)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error("PDF not found: " + str(args.pdf))

    conn = init_database()

    existing = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    if existing == 0:
        documents = extract_pages(args.pdf)
        nodes = semantic_chunk(documents)
        count = save_nodes(conn, nodes)
        print(f"Stored {count} semantic nodes")
    else:
        print(f"Resuming {existing} stored nodes")

    generate_all(conn, args.model, args.pairs, args.limit)

    count = conn.execute("SELECT COUNT(*) FROM qa").fetchone()[0]
    print(f"Total QA pairs: {count}")

    export_dataset(conn, args.output)
    conn.close()


if __name__ == "__main__":
    # subprocess.run(['python', 'sft_datagen.py', '--pdf', 'data/7110.65BB_Bsc_w_Chg_1_2_and_3_dtd_7-9-26_Final.pdf'])
    main()
