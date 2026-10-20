from local_jev import LocalJevEngine
from tabulate import tabulate

def main():
    engine = LocalJevEngine()

    # Define enterprise business queries
    tickets = [
        {
            "id": "TICK-001",
            "text": "UPI payment succeeded, money deducted from HDFC account, but booking status shows failed! Please refund or confirm urgently."
        },
        {
            "id": "TICK-002",
            "text": "Can we get custom pricing or volume licensing if our enterprise team buys 500+ developer seats?"
        },
        {
            "id": "TICK-003",
            "text": "The web application dashboard is throwing a 502 Bad Gateway error when downloading CSV reports."
        }
    ]

    # Define the 3 Jev Primitives
    questions_schema = {
        "department_routing": {
            "type": "choice",
            "instructions": "This customer request belongs to",
            "criteria": {
                "billing and payment": "Invoices, transaction failures, refunds",
                "sales and enterprise": "Bulk discounts, business quotes, subscriptions",
                "technical support": "Errors, system crashes, server bugs"
            }
        },
        "urgency_rating": {
            "type": "score",
            "instructions": "Assess urgency",
            "levels": ["low", "moderate", "high", "critical"]
        },
        "escalate_to_manager": {
            "type": "noul",
            "instructions": "Does this issue involve financial loss or system outage requiring immediate escalation?"
        }
    }

    table_data = []

    print("Running Decision Model Pipeline...\n")
    for t in tickets:
        res = engine.decide(t["text"], questions_schema)
        d = res["decisions"]
        
        table_data.append([
            t["id"],
            t["text"][:40] + "...",
            d["department_routing"]["winner"],
            f"{d['department_routing']['confidence'] * 100:.1f}%",
            f"{d['urgency_rating']['calculated_score']} / 4.0",
            "YES" if d["escalate_to_manager"]["verdict"] else "NO",
            f"{res['latency_ms']} ms"
        ])

    headers = ["Ticket ID", "Query Preview", "Routing (Choice)", "Confidence", "Urgency (Score)", "Escalate (Noul)", "Latency"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

if __name__ == "__main__":
    main()