import time
import json
import csv
import numpy as np
from local_jev import LocalJevEngine

def run_benchmarks():
    print("=" * 60)
    print("🚀 Initializing Local Jev System-1 Engine Benchmark...")
    print("=" * 60)
    
    t0 = time.time()
    engine = LocalJevEngine()
    load_time = (time.time() - t0) * 1000
    print(f"✅ Model loaded into memory in {load_time:.2f} ms\n")

    # Enterprise Test Dataset (10 Real-World Edge Cases)
    test_cases = [
        {
            "id": "TC-01",
            "text": "UPI payment succeeded, ₹4,999 deducted from bank account, but booking shows failed!",
            "expected_dept": "billing and payment",
            "expect_escalation": True
        },
        {
            "id": "TC-02",
            "text": "Looking for volume enterprise licensing, custom SLA terms, and pricing for a 250-seat team.",
            "expected_dept": "sales and enterprise",
            "expect_escalation": False
        },
        {
            "id": "TC-03",
            "text": "Critical production incident: 502 Bad Gateway across all authentication endpoints for past 20 minutes.",
            "expected_dept": "technical support",
            "expect_escalation": True
        },
        {
            "id": "TC-04",
            "text": "Where can I find the export button to download last month's PDF invoice?",
            "expected_dept": "billing and payment",
            "expect_escalation": False
        },
        {
            "id": "TC-05",
            "text": "Can someone help me schedule a product demo with your technical sales architect?",
            "expected_dept": "sales and enterprise",
            "expect_escalation": False
        },
        {
            "id": "TC-06",
            "text": "Getting a 401 Unauthorized error when hitting the /api/v2/orders endpoint using valid bearer token.",
            "expected_dept": "technical support",
            "expect_escalation": False
        },
        {
            "id": "TC-07",
            "text": "You charged my credit card twice for the annual plan renewal. Reverse the duplicate charge immediately.",
            "expected_dept": "billing and payment",
            "expect_escalation": True
        },
        {
            "id": "TC-08",
            "text": "We are a Fortune 500 company looking to replace our legacy vendor with your enterprise plan.",
            "expected_dept": "sales and enterprise",
            "expect_escalation": False
        },
        {
            "id": "TC-09",
            "text": "Our database connection pool is completely exhausted, entire production cluster is frozen.",
            "expected_dept": "technical support",
            "expect_escalation": True
        },
        {
            "id": "TC-10",
            "text": "Could you please update my registered billing email address from old@domain.com to new@domain.com?",
            "expected_dept": "billing and payment",
            "expect_escalation": False
        }
    ]

    schema = {
        "department_routing": {
            "type": "choice",
            "instructions": "This enterprise request belongs to the category of",
            "criteria": {
                "billing and payment": "Payment failures, refunds, invoice requests, and transaction disputes.",
                "sales and enterprise": "Pricing quotes, volume licensing, partnership inquiries, and demos.",
                "technical support": "System crashes, bug reports, API timeouts, and 502 gateway errors."
            }
        },
        "urgency_rating": {
            "type": "score",
            "instructions": "Evaluate the customer impact and business urgency level",
            "levels": ["low", "moderate", "high", "critical"]
        },
        "escalate_to_manager": {
            "type": "noul",
            "instructions": "Does this issue involve financial loss, legal risk, or severe downtime requiring immediate human intervention?"
        }
    }

    results = []
    latencies = []
    correct_routes = 0

    print(f"Running evaluation across {len(test_cases)} enterprise scenarios...\n")
    print(f"{'ID':<6} | {'Route':<22} | {'Conf':<6} | {'Score':<6} | {'Escalate':<8} | {'Latency':<8}")
    print("-" * 68)

    for tc in test_cases:
        res = engine.decide(tc["text"], schema)
        decisions = res["decisions"]
        lat = res["latency_ms"]
        latencies.append(lat)

        pred_route = decisions["department_routing"]["winner"]
        conf = decisions["department_routing"]["confidence"]
        score = decisions["urgency_rating"]["calculated_score"]
        escalate = decisions["escalate_to_manager"]["verdict"]

        if pred_route.lower() == tc["expected_dept"].lower():
            correct_routes += 1

        print(f"{tc['id']:<6} | {pred_route[:20]:<22} | {conf*100:>4.1f}% | {score:>4.2f}/4 | {str(escalate):<8} | {lat:>5.1f} ms")

        results.append({
            "test_id": tc["id"],
            "query": tc["text"],
            "predicted_department": pred_route,
            "confidence": f"{conf:.4f}",
            "urgency_score": f"{score:.2f}",
            "escalated": escalate,
            "latency_ms": f"{lat:.2f}"
        })

    # Save to CSV
    csv_file = "benchmark_results.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Compute Stats
    accuracy = (correct_routes / len(test_cases)) * 100
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)

    print("-" * 68)
    print("\n📊 BENCHMARK SUMMARY REPORT:")
    print(f"  • Total Test Cases      : {len(test_cases)}")
    print(f"  • Routing Accuracy       : {accuracy:.1f}% ({correct_routes}/{len(test_cases)})")
    print(f"  • Mean Latency          : {mean_lat:.2f} ms")
    print(f"  • Median Latency (p50)  : {median_lat:.2f} ms")
    print(f"  • 95th Percentile (p95) : {p95_lat:.2f} ms")
    print(f"  • Min / Max Latency     : {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"\n📁 Raw dataset exported to: {csv_file}\n")

if __name__ == "__main__":
    run_benchmarks()