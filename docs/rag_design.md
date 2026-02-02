# RAG Design (Structured Data)

## Knowledge base
The table `marketing_kpi` contains:
- Campaign identifiers
- Performance KPIs (CTR, CVR, CPL, ROI)
- Revenue and spend
- Segmentation labels

This table acts as a **structured knowledge base**.

## Retrieval
When a user asks:
“Which campaigns are underperforming?”

The system retrieves:
- Campaigns with low ROI
- High CPL
- Low CVR

using SQL queries on `marketing_kpi`.

## Generation
The retrieved rows are provided to a Large Language Model
to generate:
- Natural language explanations
- Comparisons
- Recommendations

This implements a **RAG on structured data**.

## Why this works
Unlike PDF-based RAG, this system:
- Uses live, computed KPIs
- Is always up to date
- Produces verifiable answers based on metrics
