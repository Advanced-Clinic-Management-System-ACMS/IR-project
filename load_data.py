# load_lotte.py
import ir_datasets

print("=" * 60)
print("📚 تحميل مجموعة LoTTE Lifestyle Forum")
print("=" * 60)

# تحميل الوثائق والاستعلامات معاً
dataset = ir_datasets.load("lotte/lifestyle/dev/forum")

print(f"\n📄 عدد الوثائق: {dataset.docs_count():,}")
print(f"❓ عدد الاستعلامات: {dataset.queries_count()}")
print(f"⭐ عدد التقييمات (qrels): {dataset.qrels_count():,}")

# عرض أول وثيقة
print("\n📄 أول وثيقة:")
for doc in dataset.docs_iter():
    print(f"   ID: {doc.doc_id}")
    print(f"   النص: {doc.text[:200]}...")
    break

# عرض أول استعلام
print("\n❓ أول استعلام:")
for query in dataset.queries_iter():
    print(f"   ID: {query.query_id}")
    print(f"   السؤال: {query.text}")
    break

# عرض أول تقييم
print("\n⭐ أول تقييم (qrel):")
for qrel in dataset.qrels_iter():
    print(f"   Query {qrel.query_id} → Document {qrel.doc_id} (Relevance: {qrel.relevance})")
    break

print("\n" + "=" * 60)
print("✅ تم التحميل بنجاح! المجموعة جاهزة للمشروع.")
print("=" * 60)