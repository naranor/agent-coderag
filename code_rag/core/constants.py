"""
Central location for project-wide constants.
"""

# Embedding Model Constants
EMBEDDING_DIM = 384
MAX_TOKEN_LENGTH = 512
PAD_ID = 0
PAD_TOKEN = "[PAD]"  # nosec B105

# Performance & Timeout Constants
DEFAULT_SUBPROCESS_TIMEOUT = 30
METADATA_FETCH_TIMEOUT = 60
LLM_REQUEST_TIMEOUT = 30
MAX_CONCURRENT_TASKS = 10

# Database Constants
RELATIONS_BATCH_SIZE = 100
