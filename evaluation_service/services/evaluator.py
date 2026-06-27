# The core logic looping through the data with multithreading
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluation_service.core.metrics import average_precision, ndcg_at_k
from evaluation_service.schemas.payloads import EvaluateRequest, ModelMetrics

class EvaluationService:
    @staticmethod
    def generate_report(payload: EvaluateRequest) -> dict[str, ModelMetrics]:
        report: dict[str, ModelMetrics] = {}
        
        # تشغيل تقييم كل نموذج بالتوازي
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for model_name, run in payload.runs.items():
                # إرسال كل نموذج إلى Thread منفصل
                future = executor.submit(
                    EvaluationService._evaluate_model,
                    run,
                    payload.qrels,
                    payload.k
                )
                futures[future] = model_name
            
            # جمع النتائج عند انتهاء كل Thread
            for future in as_completed(futures):
                model_name = futures[future]
                report[model_name] = future.result()
        
        return report

    @staticmethod
    def _evaluate_model(
        run: dict[str, list[str]],
        qrels: dict[str, dict[str, int]],
        k: int
    ) -> ModelMetrics:
        """تقييم نموذج واحد (يعمل في Thread منفصل)"""
        ap_values: list[float] = []
        recall_values: list[float] = []
        precision_values: list[float] = []
        ndcg_values: list[float] = []

        for query_id, ranked_docs in run.items():
            relevant = qrels.get(query_id, {})
            rel_set = {doc_id for doc_id, rel in relevant.items() if rel > 0}
            
            if not rel_set:
                continue

            ranked = ranked_docs[:k]
            binary_relevances = [1 if doc_id in rel_set else 0 for doc_id in ranked]

            ap_values.append(average_precision(binary_relevances))
            recall_values.append(len(set(ranked) & rel_set) / len(rel_set))
            precision_values.append(sum(binary_relevances) / k)
            
            ndcg_values.append(
                ndcg_at_k([relevant.get(doc_id, 0) for doc_id in ranked], k)
            )

        return ModelMetrics(
            MAP=round(sum(ap_values) / len(ap_values), 4) if ap_values else 0.0,
            Recall=round(sum(recall_values) / len(recall_values), 4) if recall_values else 0.0,
            PrecisionAt10=round(sum(precision_values) / len(precision_values), 4) if precision_values else 0.0,
            nDCGAt10=round(sum(ndcg_values) / len(ndcg_values), 4) if ndcg_values else 0.0,
        )