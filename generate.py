import os
import json
import uuid
import time
import warnings
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Generator, List, Any, Union
from pydantic import BaseModel
from urllib.parse import quote

import boto3
import qdrant_client
import nest_asyncio
from tavily import TavilyClient
from botocore.exceptions import ClientError
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core.schema import QueryBundle, Node
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

warnings.filterwarnings("ignore")

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nest_asyncio.apply()

logger = logging.getLogger(__name__)


class PromptConfig(BaseModel):
    """Configuration for prompt templates."""

    citation_template: str = "citation_template.prompt"
    qa_template: str = "qa_template.prompt"
    refine_template: str = "refine_template.prompt"
    conv_title_template: str = "conversation_title.prompt"
    related_queries_template: str = "related_queries.prompt"
    system_prompt: str = "system.prompt"
    tavily_template: str = "tavily.prompt"
    greeting_classifier: str = "greeting_classifier.prompt"
    greeting: str = "greeting.prompt"


class StorageManager:
    """Manages S3 storage operations including persist directory and chat history."""

    _s3_client = None
    _chat_hist = None
    _chat_id = None
    _persist_dir_cache = {}

    def __init__(self, config: Dict[str, Any], secret: Dict[str, Any]):
        self._config = config
        self._secret = secret
        self._init_s3_client()

    def _init_s3_client(self) -> boto3.client:
        """Initialize S3 client with credentials."""
        if StorageManager._s3_client is None:
            try:
                logger.info("Initializing S3 client...")
                StorageManager._s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self._secret["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=self._secret["AWS_SECRET_ACCESS_KEY"],
                    region_name=self._config["AWS_REGION"],
                )
            except ClientError as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error initializing S3 client: {e}")
                raise

    def load_persist_dir(self, persist_dir: str, collection_name: str) -> str:
        cache_key = f"{persist_dir}/{collection_name}"
        if cache_key in StorageManager._persist_dir_cache:
            logger.info(
                f"Using cached persist directory: {StorageManager._persist_dir_cache[cache_key]}"
            )
            return StorageManager._persist_dir_cache[cache_key]

        try:
            logger.info(f"Loading persist directory for collection: {collection_name}")
            s3_persist_path = f"{persist_dir}/{collection_name}"

            s3_response = StorageManager._s3_client.list_objects_v2(
                Bucket=self._config["S3_PERSIST_BUCKET"], Prefix=s3_persist_path
            )

            if "Contents" not in s3_response:
                logger.error(f"Persist directory not found: {s3_persist_path}")
                raise ValueError(f"Persist directory {s3_persist_path} not found")

            local_persist_dir = Path(
                self._config["S3_PERSIST_DIR"], persist_dir, collection_name
            )
            local_persist_dir.mkdir(parents=True, exist_ok=True)

            for obj in s3_response["Contents"]:
                if obj["Key"].endswith("/"):
                    continue

                file_path = Path(self._config["S3_PERSIST_DIR"], obj["Key"])
                if file_path.exists():
                    logger.info(f"Deleting existing file: {file_path}")
                    file_path.unlink()

                logger.info(f"Downloading file: {file_path}")
                file_path.parent.mkdir(parents=True, exist_ok=True)

                StorageManager._s3_client.download_file(
                    self._config["S3_PERSIST_BUCKET"], obj["Key"], str(file_path)
                )

            logger.info(f"Successfully loaded persist directory to {local_persist_dir}")
            StorageManager._persist_dir_cache[cache_key] = local_persist_dir
            return local_persist_dir

        except ClientError as e:
            logger.error(f"S3 error loading persist directory: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading persist directory: {e}")
            raise

    def load_chat_history(self, curr_chat_id: str) -> Optional[str]:
        """Load chat history for given chat ID."""
        try:
            logger.info(f"Loading chat history for chat ID: {curr_chat_id}")

            if (
                curr_chat_id == StorageManager._chat_id
                and StorageManager._chat_hist is not None
            ):
                logger.debug("Returning cached chat history")
                return StorageManager._chat_hist

            StorageManager._chat_id = curr_chat_id
            chat_summ_file = (
                f"{self._config['S3_CHAT_HISTORY']}/{StorageManager._chat_id}.md"
            )

            s3_chat_summ_obj = StorageManager._s3_client.list_objects_v2(
                Bucket=self._config["S3_LOGS_BUCKET"],
                Delimiter="/",
                Prefix=f"{self._config['S3_CHAT_HISTORY']}/",
            )

            if "Contents" not in s3_chat_summ_obj:
                logger.info("Creating new chat history directory")
                StorageManager._s3_client.put_object(
                    Bucket=self._config["S3_LOGS_BUCKET"],
                    Key=f"{self._config['S3_CHAT_HISTORY']}/",
                )
            elif chat_summ_file in [
                content["Key"] for content in s3_chat_summ_obj["Contents"]
            ]:
                logger.info("Loading existing chat history")
                response = StorageManager._s3_client.get_object(
                    Bucket=self._config["S3_LOGS_BUCKET"], Key=chat_summ_file
                )
                StorageManager._chat_hist = response["Body"].read().decode("utf-8")

        except ClientError as e:
            logger.error(f"S3 error loading chat history: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading chat history: {e}")
            raise

    @property
    def chat_hist(self) -> Optional[str]:
        """Get current chat history."""
        return self._chat_hist

    @chat_hist.setter
    def chat_hist(self, value: str):
        """Set chat history."""
        self._chat_hist = value


class PromptManager:
    _prompts = None

    @staticmethod
    def load_prompts(config: Dict[str, Any]) -> PromptConfig:
        if PromptManager._prompts is None:
            try:
                logger.info("Loading prompt templates...")
                prompts = {}

                for prompt_name, prompt_info in PromptConfig.model_fields.items():
                    with open(
                        Path(config["PROMPT_DIR"], prompt_info.default),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        prompts[prompt_name] = f.read().strip()

                logger.info("Successfully loaded all prompt templates")
                PromptManager._prompts = PromptConfig(**prompts)

            except Exception as e:
                logger.error(f"Error loading prompts: {e}")
                raise

        return PromptManager._prompts


class ModelManager:
    """Manages loading and accessing LLM and embedding models."""

    _embed_model = None
    _llm_model = None

    def __init__(self, config: Dict[str, Any], secret: Dict[str, str]):
        self._config = config
        self._secret = secret
        logger.info("Initializing ModelManager...")

    def _load_embed_model(self) -> HuggingFaceEmbedding:
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self._config['HF_EMBED']}")
            return HuggingFaceEmbedding(model_name=self._config["HF_EMBED"])
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _load_llm_model(self) -> Union[OpenAI, Ollama]:
        """Load the LLM model with retry logic."""
        try:
            model_type = self._config["LLM_MODEL_TYPE"]
            logger.info(f"Loading LLM model: {self._config[model_type]['MODEL_NAME']}")

            if model_type == "OPENAI":
                return OpenAI(
                    model=self._config["OPENAI"]["MODEL_NAME"],
                    api_key=self._secret["OPENAI_API_KEY"],
                    temperature=self._config["OPENAI"]["TEMPERATURE"],
                    top_p=self._config["OPENAI"]["TOP_P"],
                    max_tokens=self._config["OPENAI"]["MAX_TOKENS"],
                    timeout=self._config["OPENAI"]["REQUEST_TIMEOUT"],
                )
            elif model_type == "OLLAMA":
                return Ollama(
                    model=self._config["OLLAMA"]["MODEL_NAME"],
                    temperature=self._config["OLLAMA"]["TEMPERATURE"],
                    top_p=self._config["OLLAMA"]["TOP_P"],
                    request_timeout=self._config["OLLAMA"]["REQUEST_TIMEOUT"],
                    additional_kwargs={
                        "num_ctx": self._config["OLLAMA"]["CTX_LENGTH"],
                        "num_predict": self._config["OLLAMA"]["PREDICT_LENGTH"],
                        "cache": False,
                    },
                )
            elif model_type == "ANTHROPIC":
                return Anthropic(
                    model=self._config["ANTHROPIC"]["MODEL_NAME"],
                    api_key=self._secret["ANTHROPIC_API_KEY"],
                    temperature=self._config["ANTHROPIC"]["TEMPERATURE"],
                    max_tokens=self._config["ANTHROPIC"]["MAX_TOKENS"],
                    timeout=self._config["ANTHROPIC"]["REQUEST_TIMEOUT"],
                )
            else:
                raise ValueError(f"Invalid LLM model type: {model_type}")

        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            raise

    @property
    def embed_model(self) -> HuggingFaceEmbedding:
        """Get the loaded embedding model."""
        if ModelManager._embed_model is None:
            ModelManager._embed_model = self._load_embed_model()
        return ModelManager._embed_model

    @property
    def llm_model(self) -> Union[OpenAI, Ollama]:
        """Get the loaded LLM model."""
        if ModelManager._llm_model is None:
            ModelManager._llm_model = self._load_llm_model()
        return ModelManager._llm_model


class Generate:
    """Main class for generating answers to user queries."""

    def __init__(
        self,
        config: Dict[str, Any],
        secret: Dict[str, Any],
        chat_id: str,
        query: str,
        category_name: str,
        persist_dir: str,
        collection_name: str,
        s3_manager: StorageManager,
        metadata: Dict[str, Any] = {},
    ):
        self._config = config
        self._secret = secret
        self._query = query
        self._category_name = category_name
        self._storage_manager = s3_manager
        self._tavily_client = TavilyClient(api_key=self._secret["TAVILY_API_KEY"])

        logger.info(f"Initializing Generate for category: {category_name}")

        self._prompts = PromptManager.load_prompts(self._config)

        self._model_manager = ModelManager(config, self._secret)
        Settings.llm = self._model_manager.llm_model
        Settings.embed_model = self._model_manager.embed_model

        # Load chat history and prepare query
        self._storage_manager.load_chat_history(chat_id)
        self._refined_query = self._prepare_query()

        # Setup query engine
        self._persist = self._storage_manager.load_persist_dir(
            persist_dir, collection_name
        )
        index = self._setup_storage_context(collection_name)
        metadata_filters = self._prepare_metadata_filters(metadata)
        self._init_query_engine(index, metadata_filters)

    def _prepare_query(self) -> str:
        """Prepare the refined query with chat history."""
        if self._storage_manager.chat_hist is not None:
            return f"<|CHAT HISTORY|>: {self._storage_manager.chat_hist}\n\n<|QUERY|>: {self._query}"
        return f"<|QUERY|>: {self._query}"

    def _setup_storage_context(self, collection_name: str) -> StorageContext:
        """Setup storage context with Qdrant vector store."""
        try:
            client = qdrant_client.QdrantClient(
                url=self._secret["QDRANT_URL"], api_key=self._secret["QDRANT_API_KEY"]
            )

            aclient = qdrant_client.AsyncQdrantClient(
                url=self._secret["QDRANT_URL"], api_key=self._secret["QDRANT_API_KEY"]
            )

            vector_store = QdrantVectorStore(
                client=client,
                aclient=aclient,
                collection_name=collection_name,
                enable_hybrid=self._config["QDRANT_ENABLE_HYBRID"],
                fastembed_sparse_model=self._config["FASTEMBED_SPARSE_MODEL"],
                prefer_grpc=False,
            )

            storage_context = StorageContext.from_defaults(
                persist_dir=self._persist, vector_store=vector_store
            )
            index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            logger.error(f"Error setting up storage context: {e}")
            raise

    def _prepare_metadata_filters(
        self, metadata: Dict[str, Any]
    ) -> List[MetadataFilter]:
        """Prepare metadata filters from metadata dict."""
        logger.info(f"Preparing metadata filters: {metadata}")
        return [MetadataFilter(key=key, value=value) for key, value in metadata.items()]

    def _init_query_engine(
        self, index, metadata_filters: Optional[List[MetadataFilter]]
    ) -> None:
        """Initialize the citation query engine."""
        try:
            sim_processor = SimilarityPostprocessor(
                similarity_cutoff=self._config["RAG_SIMILARITY_CUTOFF"]
            )
            rerank = CohereRerank(
                api_key=self._model_manager._secret["COHERE_API_KEY"],
                model=self._config["COHERE_RERANKER"],
                top_n=self._config["RAG_RERANKED_TOP_N"],
            )

            self.query_engine = CitationQueryEngine.from_args(
                index,
                embed_model=self._model_manager.embed_model,
                chat_mode="context",
                citation_chunk_size=self._config["RAG_CITATION_CHUNK_SIZE"],
                citation_chunk_overlap=self._config["RAG_CITATION_CHUNK_OVERLAP"],
                citation_qa_template=PromptTemplate(
                    self._prompts.citation_template + self._prompts.qa_template
                ),
                citation_refine_template=PromptTemplate(
                    self._prompts.citation_template + self._prompts.refine_template
                ),
                similarity_top_k=self._config["RAG_SIMILARITY_TOP_K"],
                node_postprocessors=[rerank, sim_processor],
                filters=MetadataFilters(filters=metadata_filters or []),
                llm=self._model_manager.llm_model,
                streaming=self._config["RAG_STREAMING"],
            )
            logger.info("Successfully initialized query engine")

        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            raise

    def generate_answer(self) -> Generator[str, None, None]:
        """Generate and yield the answer for the given query."""
        try:
            logger.debug("Checking if query is a greeting")
            is_greeting = self._model_manager.llm_model.complete(
                self._prompts.greeting_classifier.format(query=self._query)
            ).text.strip()

            if is_greeting == "True":
                logger.info("Query classified as greeting")
                greeting_prompt = [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=self._prompts.system_prompt,
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=self._prompts.greeting.format(query=self._query),
                    ),
                ]
                greeting_response = self._model_manager.llm_model.stream_chat(
                    greeting_prompt
                )

                for text in greeting_response:
                    yield json.dumps(
                        {
                            "response_id": str(uuid.uuid4()),
                            "type": "greeting",
                            "text": text.delta,
                        }
                    )

                return

            answer = ""
            logger.info("Retrieving relevant documents...")
            retrieved_docs = self.query_engine.retrieve(
                QueryBundle(query_str=self._refined_query)
            )

            logger.info(f"Retrieved documents: {retrieved_docs}")
            logger.info(
                f"Score of retrieved docs: {[doc.score for doc in retrieved_docs]}"
            )
            for idx, doc in enumerate(retrieved_docs):
                logger.info(f"Document {idx+1}: {doc.node.get_text()}")

            if (
                not retrieved_docs
                or max([doc.score for doc in retrieved_docs])
                <= self._config["RAG_SIMILARITY_CUTOFF"]
            ):
                logger.warning("No relevant contexts retrieved")
                search_results = self._tavily_client.search(
                    self._query, max_results=3, search_depth="advanced"
                )["results"]
                content = "\n\n".join(
                    [
                        f"{idx+1}. Title: {result['title']}\nContent: {result['content']}\nURL: {result['url']}"
                        for idx, result in enumerate(search_results)
                    ]
                )
                tavily_prompt = [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=self._prompts.system_prompt,
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=self._prompts.tavily_template.format(
                            search_results=content, query=self._query
                        ),
                    ),
                ]

                tavily_resp = self._model_manager.llm_model.stream_chat(tavily_prompt)

                for text in tavily_resp:
                    yield json.dumps(
                        {
                            "response_id": str(uuid.uuid4()),
                            "type": "tokens",
                            "text": text.delta,
                        }
                    )

                return

            # Generate response
            logger.info("Generating response...")
            start_response = time.perf_counter()
            response = self.query_engine.query(self._refined_query)
            end_response = time.perf_counter()
            logger.info(
                f"Time taken to generate response: {end_response - start_response} seconds"
            )

            for text in response.response_gen:
                if text != "Empty Response":
                    answer += text
                    yield json.dumps(
                        {
                            "response_id": str(uuid.uuid4()),
                            "type": "tokens",
                            "text": text,
                        }
                    )

            # Process contexts and citations
            logger.info("Processing contexts and citations...")
            contexts, answer = self._process_contexts(
                answer, response.source_nodes, retrieved_docs
            )
            yield json.dumps(
                {"response_id": str(uuid.uuid4()), "type": "answer", "text": answer}
            )

            yield json.dumps(
                {
                    "response_id": str(uuid.uuid4()),
                    "type": "context",
                    "text": json.dumps(contexts),
                }
            )

            # Generate related queries
            logger.info("Generating related queries...")
            related_queries = self._model_manager.llm_model.complete(
                self._prompts.related_queries_template.format(
                    query=self._query,
                    sources="\n\n".join(
                        doc.node.get_text() for doc in response.source_nodes
                    ),
                    answer=answer,
                )
            ).text.strip()

            yield json.dumps(
                {
                    "response_id": str(uuid.uuid4()),
                    "type": "related",
                    "text": related_queries,
                }
            )

            # Generate conversation title if no chat history
            if self._storage_manager.chat_hist is None:
                logger.info("Generating conversation title...")
                conversation_title = self._model_manager.llm_model.complete(
                    self._prompts.conv_title_template.format(
                        query=self._query, category=self._category_name
                    )
                ).text.strip()

                yield json.dumps(
                    {
                        "response_id": str(uuid.uuid4()),
                        "type": "title",
                        "text": conversation_title,
                    }
                )

            # Update chat history
            self._storage_manager.chat_hist = f"{self._refined_query}\n{answer}\n\n"
            logger.info("Successfully completed response generation")

        except Exception as e:
            logger.critical(f"Error generating answer: {e}")
            raise

    def _process_contexts(
        self, answer: str, source_nodes: List[Node], retrieved_docs: List[Node]
    ) -> Dict[str, Dict[str, Any]]:
        """Process and format context information from retrieved documents."""
        try:
            logger.debug("Processing context information...")
            extract_pattern = r"^Source \d+:\s*\n"
            cited_nums = re.findall(r"\[(\d+)\]", answer)
            source_lst = []

            for source in source_nodes:
                source_text = re.sub(
                    extract_pattern, "", source.node.get_text(), flags=re.MULTILINE
                ).strip()
                source_lst.append(source_text)

            contexts = {}
            retrieved_counter = 0

            for idx, doc in enumerate(retrieved_docs):
                if str(idx + 1) not in cited_nums:
                    continue

                if doc.text.strip() in source_lst:
                    retrieved_counter += 1
                    contexts[str(retrieved_counter)] = {
                        "file_name": doc.metadata["file_name"],
                        "page_num": doc.metadata["page_num"],
                        "chunk": doc.metadata["highlighted_chunk"],
                    }
                    answer = answer.replace(
                        f"[{str(idx+1)}]",
                        f'[[{retrieved_counter}]]({self._config["PDF_BASE_URL"]}{quote(doc.metadata["file_name"])}.pdf)',
                    )

            logger.debug(f"Processed {retrieved_counter} context citations")
            return contexts, answer

        except Exception as e:
            logger.error(f"Error processing contexts: {e}")
            raise
