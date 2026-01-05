# Blog 2: Deep Dive into My Agentic RAG Implementation: Architecture and Code Patterns

## From Theory to Implementation: How A2A Actually Works in Practice

Understanding the theory of A2A protocol is one thing. Seeing how I implemented it in my 2025 Agentic RAG system is another. Let's dig into the architecture and examine the actual patterns that make my system work.


## High-Level Architecture

My Agentic RAG system has 5 layers: 

1. FastAPI application layer (for public-facing user interface)
2. Orchestrator layer (orchestrates all agents)
3. Agent execution layer (each individual agent is a separate entity; containerized with Ollama for LLM serving, etc.)
4. Data layer (Oracle Database 26ai, acts as the vector store for all the document vector embeddings)
5. LLM Backend Layer (each agent states in its Agent Card which LLM Backend it is using).

### An example Agent Card

```json
{
  "agent_id": "planner-agent-v1",
  "display_name": "Query Planner",
  "description": "Analyzes user queries and breaks them into research tasks",
  "capabilities": [
    {
      "capability_id": "analyze-query",
      "description": "Analyzes a natural language query and produces a task plan",
      "input_schema": {
        "query": "string",
        "context": "object"
      },
      "output_schema": {
        "tasks": "array",
        "reasoning": "string"
      }
    }
  ],
  "endpoints": {
    "base_url": "https://planner-agents.internal:8000",
    "task_create": "/v1/tasks",
    "task_status": "/v1/tasks/{task_id}"
  },
  "authentication": {
    "scheme": "bearer",
    "token_endpoint": "https://auth.internal/oauth/token"
  },
  "metadata": {
    "llm_backend": "ollama://local:11434",
    "model_name": "mistral:7b",
    "max_concurrent_tasks": 20,
    "response_time_sla_ms": 2000
  }
}
```

This single JSON document contains everything needed to determine how to reach the agent, how to authenticate with the agent, and even some operational details such as max concurrent tasks, and expected response time for the agent.

The task lifecycle in my system includes the following steps:

**Step 1 - Task Creation**
User submits a query through the FastAPI server. The Orchestrator creates a root task (say `task-abc123`) and sends it to the planner agent through A2A.

```sql
POST /v1/tasks
{
  "task_id": "task-abc123",
  "agent_capability": "analyze-query",
  "payload": {
    "query": "What are the main risks in our financial reports?",
    "context": { "user_id": "user-456", "session": "sess-789" }
  }
}
```

**Step 2 - Agent Processing**
The planner agent receives the task. It processes the query using its LLM backend (an Ollama instance running a Mistral model). The agent produces a structured plan ("I need to extract risks, categorize them, and rank by severity").

**Step 3 - Sub-Task Delegation**
The planner agent produces child tasks for researchers. It discovers available researcher agents using A2A discovery and sends them work through the A2A protocol.

```sql
POST /v1/tasks
{
  "task_id": "task-abc123-research-1",
  "parent_task_id": "task-abc123",
  "agent_capability": "retrieve-from-documents",
  "payload": {
    "query": "financial risks",
    "document_types": ["financial_reports"],
    "max_results": 10
  }
}
```

**Step 4 - Researcher Execution**
Each researcher agent searches the vector store (the Oracle Database 26ai) for relevant content. They are independent so if one is slow, others continue. Results are pushed back as soon as discovered.

**Step 5 - Status Monitoring**
All communication between the agents happen through the A2A protocol. The tasks have status endpoints to allow the Orchestrator to poll (or receive push notifications via Server-Sent Events) and monitor progress.

```sql
GET /v1/tasks/task-abc123/status
{
  "status": "running",
  "progress": 0.65,
  "sub_tasks": [
    { "task_id": "task-abc123-research-1", "status": "completed", "result_count": 8 },
    { "task_id": "task-abc123-research-2", "status": "running", "progress": 0.4 }
  ]
}
```

**Step 6 - Synthesis**
Once research is done, the Orchestrator sends aggregated results to the Synthesizer agent. The Synthesizer agent then uses its LLM to produce a coherent response based on the retrieved documents.

Practical scaling implications:
- **Planner agents**: 2-3 instances (lightweight, fast)
- **Researcher agents**: 20+ instances (the heavy-lifters)
- **Synthesizer agents**: 5-10 instances (dependent on response generation load)

Scaling can be done independently for each agent type depending on queue-depth, response-time, or policy.

Database integration: vector-store as source of truth
Agents query the vector-store using standardized interfaces:

```sql
SELECT TOP 10
  document_id,
  chunk_text,
  TO_NUMBER(vector_distance(embedding, vector_embed(query_text, 'model_name'))) AS relevance
FROM document_chunks
WHERE user_has_access(USER_ID, document_id) -- Security trimming
ORDER BY relevance
```

Security trimming occurs at query-time not at retrieval time. Different users see different results based on their access rights.

Monitoring & Observability
Agents expose standard metrics through the A2A protocol:

```sql
/v1/agents/{agent_id}/metrics
{
  "tasks_total": 10542,
  "tasks_completed": 10521,
  "tasks_failed": 12,
  "tasks_running": 9,
  "avg_response_time_ms": 1240,
  "p95_response_time_ms": 3100,
  "queue_depth": 15,
  "last_error": "timeout on vector search"
}
```

The Orchestrator collects the metrics and makes routing decisions based on them. For example, if the average response time of a researcher agent exceeds its SLA, the Orchestrator can redirect new tasks to other agents. If there are many queued tasks in an agent, automatic scaling can kick-in.

Why this matters for your production systems:
This architecture solves very practical problems:

* Multi-tenancy: Data is separated within the database by tenant security trimming and agents do not need to know anything about multi-tenancy.
* HA: Agents are stateless. If an agent fails, another agent will pick up the task without losing any data. No complicated fail-over logic.
* Operational debugging: When a customer reports slow responses, you can track the task through the entire system and see which agent was slow, see its metrics and identify the actual issue.
* Cost optimization: By right sizing compute for each agent type, you are not paying too much for unused capacity. Scale is fine-grained and responsive.

Looking at code patterns
In my actual implementation, I use patterns like:

1. **Async Task Handlers** - Each agent has async message handlers that run in parallel to handle A2A protocol messages.
2. **Result Aggregation** - The Orchestrator uses async-await pattern to collect results from multiple agents.
3. **Gracful degradation** - If a researcher agent becomes unavailable, the Orchestrator will retry the task or use an alternate agent.
4. **Observability hooks** - Every API boundary is logged to provide traces.

These are complex architectures but the patterns are reusable. If you are building multi-agent systems, these patterns should be adopted.

The takeaway
My 2025 Agentic RAG implementation is not simply implementing A2A protocol on top of last years system. It is fundamentally changing the way agent architectures are approached from monolithic pipeline-based architectures to distributed, independently scalable services. Understanding these patterns will help you implement production-grade agentic systems that can scale.