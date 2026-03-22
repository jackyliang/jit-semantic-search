"""Synthetic benchmark dataset for evaluating JIT search strategies.

Generates realistic support-ticket-like documents with known semantic clusters
so we can measure retrieval quality against ground truth.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

# Semantic clusters: each has a theme, representative queries, and document templates.
# Ground truth: documents in a cluster are relevant to that cluster's queries.
CLUSTERS = [
    {
        "id": "onboarding",
        "queries": [
            "customer struggling with initial setup",
            "user can't complete onboarding flow",
            "new account activation problems",
        ],
        "templates": [
            "I just signed up but I'm stuck on the onboarding wizard. It keeps spinning on step {n}.",
            "New customer here. I can't figure out how to set up my account. The activation email never arrived.",
            "Having trouble getting started. The setup process asks for information I don't have yet.",
            "First time user — the onboarding tutorial crashed halfway through and now I can't restart it.",
            "I created my account yesterday but the welcome flow won't load. Just shows a blank page.",
            "Trying to complete initial configuration but the 'Next' button is grayed out on the profile step.",
            "The getting-started guide references features I can't find in my dashboard. Am I missing something?",
            "Just purchased a subscription but the account provisioning seems stuck. No access to any features.",
        ],
    },
    {
        "id": "billing",
        "queries": [
            "customer charged incorrectly",
            "billing dispute unexpected charge",
            "refund request for wrong amount",
        ],
        "templates": [
            "I was charged ${amount} but my plan is only ${lower}. Please explain this discrepancy.",
            "My credit card was billed twice this month. I need a refund for the duplicate charge.",
            "I cancelled my subscription last week but I was still charged today. Transaction ID: {txn}.",
            "The invoice shows a charge for a feature I never enabled. Please credit my account.",
            "I downgraded my plan 3 days ago but the new billing amount isn't reflected yet.",
            "Unexpected charge of ${amount} appeared on my statement. I didn't authorize any upgrades.",
            "I need an itemized receipt for my last payment. The current invoice is missing line items.",
            "Our finance team flagged an overcharge on our enterprise account. Need this resolved ASAP.",
        ],
    },
    {
        "id": "performance",
        "queries": [
            "application running very slowly",
            "dashboard takes forever to load",
            "API response times are terrible",
        ],
        "templates": [
            "The dashboard has been extremely slow since this morning. Pages take 30+ seconds to load.",
            "API response times went from 200ms to 5 seconds overnight. Is there a known issue?",
            "Our users are complaining about sluggish performance. The app is basically unusable right now.",
            "Load times for the reporting module have degraded significantly after the last update.",
            "Getting timeout errors when trying to export large datasets. Used to work fine last week.",
            "The search feature is painfully slow. Takes over 10 seconds to return any results.",
            "Real-time sync has stopped being real-time. There's a 30-second delay on all updates now.",
            "Our integration endpoint is hitting rate limits even though we're well under our quota.",
        ],
    },
    {
        "id": "data_loss",
        "queries": [
            "customer lost their data",
            "files disappeared from account",
            "records deleted unexpectedly",
        ],
        "templates": [
            "All my saved reports are gone. I had 50+ custom reports that vanished overnight.",
            "Several contacts were deleted from our CRM without anyone on our team doing it.",
            "I uploaded a batch of {n} records yesterday and they've all disappeared today.",
            "Our project history was wiped clean. Months of data just gone. We need this recovered.",
            "The migration tool lost half our data. Only {n} of {total} records made it across.",
            "Files I uploaded to the document center are showing as 'not found' now.",
            "My entire configuration was reset to defaults. All custom settings are gone.",
            "We had a shared workspace with {n} team members' data. Everything is now empty.",
        ],
    },
    {
        "id": "integration",
        "queries": [
            "third party integration broken",
            "API webhook not firing",
            "Slack integration stopped working",
        ],
        "templates": [
            "Our Slack integration stopped sending notifications 2 days ago. No changes on our end.",
            "The webhook endpoint returns 200 but no events are being delivered to our server.",
            "Salesforce sync broke after your last update. Records aren't flowing between systems.",
            "Our custom API integration is getting 403 errors even though the API key hasn't changed.",
            "The Zapier connector can't authenticate anymore. Worked fine until this morning.",
            "JIRA integration is creating duplicate tickets. Every event triggers two webhook calls.",
            "SSO login via Okta stopped working. Users get 'provider configuration error' on login.",
            "The data export API endpoint changed without notice and broke our nightly ETL pipeline.",
        ],
    },
    {
        "id": "security",
        "queries": [
            "unauthorized access to account",
            "suspicious login activity",
            "security breach concern",
        ],
        "templates": [
            "I see login attempts from IP addresses I don't recognize. Has my account been compromised?",
            "Someone changed my account email without my permission. I need this locked down immediately.",
            "Our admin noticed a new API key was created that nobody on the team authorized.",
            "There are audit log entries showing data exports at 3am. Nobody on our team was working then.",
            "A former employee still has access to our account even though we removed them months ago.",
            "We're seeing suspicious activity — someone is making bulk API calls from an unknown location.",
            "Two-factor authentication was disabled on our account without any admin action.",
            "Our security scan flagged that your platform is transmitting data over unencrypted channels.",
        ],
    },
    {
        "id": "feature_request",
        "queries": [
            "requesting new functionality",
            "feature suggestion for improvement",
            "wish list for product updates",
        ],
        "templates": [
            "It would be great if we could schedule reports to run automatically on a weekly basis.",
            "Can you add dark mode to the dashboard? Our team works late and the bright UI is harsh.",
            "We really need bulk editing capabilities. Updating records one by one is not scalable.",
            "Please add CSV export to the analytics module. Currently we can only export as PDF.",
            "Would love to see a mobile app. Having to use the desktop site on mobile is painful.",
            "Can we get more granular role permissions? The current admin/user split is too coarse.",
            "A notification system for threshold alerts would save us hours of manual monitoring.",
            "We need an audit trail feature for compliance. Currently there's no way to track changes.",
        ],
    },
    {
        "id": "ui_bug",
        "queries": [
            "user interface display glitch",
            "buttons not working on page",
            "visual rendering issue in browser",
        ],
        "templates": [
            "The dropdown menu overlaps with the header on Chrome. Can't click any items in it.",
            "Submit button doesn't respond on Safari. Works fine on Firefox though.",
            "The chart visualization is rendering with garbled text labels on high-DPI displays.",
            "Modal dialogs are appearing behind the page overlay — can see them but can't interact.",
            "The responsive layout is completely broken on tablets. Sidebar covers the main content.",
            "After the last update, all icons are showing as empty squares. Looks like missing fonts.",
            "Date picker widget shows dates in wrong format (MM/DD vs DD/MM) based on locale settings.",
            "The drag-and-drop interface freezes the browser tab when reordering more than 10 items.",
        ],
    },
]


@dataclass
class BenchmarkDataset:
    """A benchmark dataset with documents, queries, and ground truth relevance."""
    documents: list[str]
    queries: list[str]
    # relevance[query_idx] = set of relevant doc indices
    relevance: dict[int, set[int]] = field(default_factory=dict)
    cluster_labels: list[str] = field(default_factory=list)


def generate_dataset(
    docs_per_cluster: int = 50,
    seed: int = 42,
) -> BenchmarkDataset:
    """Generate a synthetic benchmark dataset.

    Args:
        docs_per_cluster: Number of documents per semantic cluster.
        seed: Random seed for reproducibility.

    Returns:
        BenchmarkDataset with documents, queries, and ground truth relevance judgments.
    """
    rng = random.Random(seed)

    documents: list[str] = []
    queries: list[str] = []
    relevance: dict[int, set[int]] = {}
    cluster_labels: list[str] = []

    for cluster in CLUSTERS:
        cluster_start = len(documents)

        # Generate documents by sampling and filling templates
        for i in range(docs_per_cluster):
            template = rng.choice(cluster["templates"])
            # Fill in template variables
            doc = template.format(
                n=rng.randint(2, 100),
                amount=rng.randint(10, 500),
                lower=rng.randint(5, 50),
                txn=hashlib.md5(f"{cluster['id']}-{i}".encode()).hexdigest()[:12],
                total=rng.randint(100, 10000),
            )
            documents.append(doc)
            cluster_labels.append(cluster["id"])

        cluster_end = len(documents)
        relevant_indices = set(range(cluster_start, cluster_end))

        # Add queries for this cluster
        for q in cluster["queries"]:
            query_idx = len(queries)
            queries.append(q)
            relevance[query_idx] = relevant_indices

    return BenchmarkDataset(
        documents=documents,
        queries=queries,
        relevance=relevance,
        cluster_labels=cluster_labels,
    )
