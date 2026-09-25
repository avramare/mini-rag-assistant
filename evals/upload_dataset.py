"""Validate a dataset and upsert it to Langfuse (questions and metadata only; facts stay local).

`generate` uploads too, so this is only needed to inspect a dataset in Langfuse before a run.
Usage: uv run python -m evals.upload_dataset [path]   (default evals/dataset.jsonl)
"""

import sys
from pathlib import Path

from evals.dataset import DatasetError, check_facts_in_corpus, load_dataset
from evals.publish import LangfusePublisher
from evals.run_experiment import DEFAULT_DATASET


def main() -> int:
    from mini_rag.config import Settings
    from mini_rag.documents import load_documents
    from mini_rag.tracing import make_langfuse
    from mini_rag.users import load_users

    s = Settings()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    try:
        dataset = load_dataset(path, set(load_users(s.users_file)))
        check_facts_in_corpus(dataset, load_documents(s.docs_dir))
    except DatasetError as exc:
        print(f"[fail] {exc}")
        return 1
    client = make_langfuse(s)
    if client is None:
        print("[fail] Langfuse keys missing in .env")
        return 1
    publisher = LangfusePublisher(client)
    publisher.upload_dataset(dataset)
    publisher.flush()
    print(f"[ok]   {len(dataset.items)} items upserted to Langfuse dataset {dataset.langfuse_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
