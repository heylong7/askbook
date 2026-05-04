"""One-shot helper: list chunks per doc with first 80 chars to aid manual labeling.

Usage: uv run python scripts/build_seed_chunk_ids.py --collection demo
Output: prints (doc_id, chunk_id, snippet) rows; copy chunk_ids into seed_manual.yaml.
"""

from __future__ import annotations

import argparse

from askbook.config.settings import load_settings
from askbook.core.registry import ServiceRegistry


def main() -> None:
    ap = argparse.ArgumentParser(
        description="List all chunk IDs for a collection to aid seed_manual labeling"
    )
    ap.add_argument("--collection", default="demo")
    args = ap.parse_args()

    cfg = load_settings(None)
    reg = ServiceRegistry()
    embedder = reg.build_embedder(cfg.embedding)
    store = reg.build_vectorstore(cfg.vectorstore)

    full = store.make_collection_name(
        namespace=args.collection, embed_model=embedder.model_name
    )
    # Use chromadb directly to iterate chunks
    collection = store._client.get_collection(full)
    result = collection.get(include=["documents", "metadatas"])

    if result["ids"]:
        for i, chunk_id in enumerate(result["ids"]):
            doc_id = (
                result["metadatas"][i].get("source", "?")
                if result["metadatas"]
                else "?"
            )
            snippet = (result["documents"][i] or "")[:80].replace("\n", " ")
            print(f"{doc_id}\t{chunk_id}\t{snippet}")
    else:
        msg = (
            "No chunks found. "
            "Run `askbook ingest examples/docs/seed/ --collection demo` first."
        )
        print(msg)


if __name__ == "__main__":
    main()
