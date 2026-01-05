# From Flat Text to Syntax Trees: How cAST Enables Code-Aware RAG at Scale

Typically, I have treated knowledge (as an AI agent knows it) as flat chunks of information: a page from a PDF document, a snippet of text from a doc, or naively a portion of a source file; but mainly text data, and unstructured. cAST is a method which we will observe that allows efficiently storing code, files, and the relationship between these files, to be able to accurately be retrieved by rerankers during RAG (Retrieval-Augmented Generations), a type of retrieval I have previously mentioned in [this article](blog_1_medium.md) and regarding to the [agentic_rag](github.com/oracle-devrel/devrel-labs/tree/main/agentic_rag#readme) project.

For unstructured documents such as articles, blogs, etc., the above flat-chunk treatment does work relatively well. However, for all real codebases, it does not even come close.

Why? Because source code is hierarchical; i.e., there are no flat source code files, but rather source code files are structured into a hierarchy of repository, package, module, class/function, and finally statement. When a function is split in half, or a class’s __init__ method is merged with a completely unrelated set of imports, the semantic integrity required for agents to reason about the code is destroyed.

cAST was *recently* developed at Carnegie Mellon University as a solution to this problem. Instead of splitting code into flat chunks based upon lines or tokens, cAST splits code into hierarchically structured chunks of semantically meaningful units of source code, including functions, classes, modules, etc.

## How cAST Works: The Split-Then-Merge Algorithm

Here is a high-level overview of the algorithm used by cAST:

Parse the source code into an Abstract Syntax Tree (AST) using the Tree-sitter parser library — which is widely used by both GitHub and others. In addition to representing the source code as a tree of nodes (i.e., classes, methods, loops, conditionals, etc.) rather than as raw text, the AST provides the structural basis for the hierarchical chunking process.

Top-down recursive traversal of the AST to split the source code into chunks. Start with the root of the AST, and try to include as much of the source code as possible in each chunk. If a chunk exceeds the maximum allowed size, recursively split it.

Bottom-up merging of sibling nodes. The output of the recursive top-down traversal will be many very small chunks (e.g., import statements, variable assignments). These small chunks should be merged together into larger chunks to increase information density and to prevent excessive growth in the index. Each adjacent pair of small sibling nodes should be greedily merged into a single chunk to provide enough surrounding code to allow the LLM to understand the intent of the code.

Non-whitespace characters are counted for determining the size of the chunks. This allows the size of the chunks to be compared across different programming languages and different coding standards.

## Why Does cAST Matter?: The Evidence

Researchers applied cAST to several real-world codebases and saw some significant benefits:

Retrieval: On RepoEval (which tests code-completion with long intra-file contexts), cAST increased the recall at 5 by 4.3 points over fixed-size chunking.

Generation: On SWE-bench (the de facto standard for testing code-fixing capabilities), Pass@1 improved by 2.67 points.

Consistency Across Languages: Because cAST splits the code into chunks based upon structure (not line numbers), it generalizes better than fixed-size chunking (which performs poorly on different languages due to syntax differences).

Why is cAST important? As the authors noted: “When an agent finds a method through vector search, cAST will find the whole method and the entire class containing the method.” Therefore, the LLM can consider the entire scope of the code when generating completions — resulting in fewer hallucinations and more accurate completions.

## Making cAST Fit into Oracle Database 26ai

cAST is simply a chunking strategy, not a new database technology. Therefore, it fits perfectly into the architecture I’ve previously described.

Instead of storing flat-text chunks:

sql
CREATE TABLE CodeCollection (
id VARCHAR2(4000 BYTE) PRIMARY KEY,
text VARCHAR2(4000 BYTE), -- code chunk
metadata VARCHAR2(4000 BYTE),
embedding VECTOR
);

you would add structural metadata to represent the hierarchy of the AST:

sql
CREATE TABLE CodeNodes (
id VARCHAR2(4000 BYTE) PRIMARY KEY,
parent_id VARCHAR2(4000 BYTE), -- refers to the class/module containing the node
node_type VARCHAR2(50), -- 'function', 'class', 'module', etc.
language VARCHAR2(50),
path VARCHAR2(4000 BYTE), -- path to the file or module
name VARCHAR2(4000 BYTE), -- name of the symbol
code_snippet CLOB, -- actual code
metadata CLOB,
embedding VECTOR
);

Therefore, the retrieval pipeline would be:

Semantic search on fine-grained chunks: Vector search query on the individual functions/methods’ embeddings.

Structural enrichment: Once you find a match, look up the parent_id to get the enclosing class/module context.

Single-database logging: All of your agent decisions and retrievals go into A2A_EVENTS, allowing you to identify which code patterns are causing problems for the agents.

Note: This is equivalent to a parent-child retriever: Precision comes from searching on the "leaf" nodes (functions, methods), while Context is provided by returning the "parent" nodes (classes, modules) to the LLM.

## Engineering Feasibility

Is this feasible in a production environment? Yes. The authors of cAST made their implementation available as an open-source project (astchunk on GitHub), and there are now reports of multiple teams successfully implementing cAST in LangChain, LlamaIndex, and custom RAG pipelines. Additionally, Tree-sitter is a battle-tested library (GitHub, Neovim, etc.), so parsing large codebases is fast and reliable.

I plan to implement cAST into the [agentic_rag](https://github.com/oracle-devrel/devrel-labs/tree/main/agentic_rag#readme) repository, as I'm working on the implementation in [this other repository](). 

In summary, cAST makes it easier to write an implementation of code-RAG. It allows the agent to see a structured, complete view of all of the code-base, which allows the agent to reason properly about thousands of files and their relationships.