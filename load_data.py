"""LoTTE dataset verification script."""
import sys
from pathlib import Path

if sys.platform == "win32" and not sys.flags.utf8_mode:
    import os

    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ir_datasets

from shared.config import DEFAULT_IR_DATASET

print("=" * 60)
print("LoTTE Lifestyle Forum Dataset")
print("=" * 60)

dataset = ir_datasets.load(DEFAULT_IR_DATASET)

print(f"\nDataset ID: {DEFAULT_IR_DATASET}")
print(f"Documents: {dataset.docs_count():,}")
print(f"Queries: {dataset.queries_count():,}")
print(f"Qrels entries: {dataset.qrels_count():,}")

print("\nFirst document:")
for doc in dataset.docs_iter():
    print(f"  doc_id: {doc.doc_id}")
    print(f"  text: {(doc.text or '')[:200]}...")
    break

print("\nFirst query:")
for query in dataset.queries_iter():
    print(f"  query_id: {query.query_id}")
    print(f"  text: {query.text}")
    break

print("\nFirst qrel:")
for qrel in dataset.qrels_iter():
    print(f"  query_id={qrel.query_id} doc_id={qrel.doc_id} relevance={qrel.relevance}")
    break

print("\n" + "=" * 60)
print("Dataset is ready. Run the offline pipeline next.")
print("=" * 60)
