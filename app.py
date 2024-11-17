from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
import boto3
import logging
import time
import yaml
import atexit
from pathlib import Path
from botocore.exceptions import ClientError
import watchtower
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from tenacity import retry, stop_after_attempt, wait_exponential
from generate import Generate, StorageManager
from secrets_manager import get_secret

# Constants
CONFIG_PATH = Path("config/config.yaml")

logger = logging.getLogger(__name__)

start_time = time.perf_counter()


class PromptConfig(BaseModel):
    """Configuration for prompt templates."""

    greeting_classifier: str = "greeting_classifier.prompt"
    greeting: str = "greeting.prompt"
    profanity_filter: str = "profanity_filter.prompt"
    history_summarizer: str = "history_summarizer.prompt"
    rephrased_query: str = "rephrased_query.prompt"


class CategoryConfig(BaseModel):
    """Configuration for category names."""

    oil_gas: str = "Oil and Gas"
    finance: str = "Finance"
    healthcare: str = "Healthcare"


class RAG(BaseModel):
    """Data model for RAG API requests with validation."""

    chat_id: str = Field(default="zpf87cm9", description="Chat ID")
    query: str = Field(..., description="User query")
    category: str = Field(..., description="Category Name")
    file_name: Optional[str] = Field(default="", description="File name")
    collection_name: Optional[str] = Field(
        default="rag_llm", description="Collection name"
    )
    persist_dir: Optional[str] = Field(
        default="persist", description="Persistent directory"
    )

    @field_validator("chat_id")
    def validate_chat_id(cls, value):
        """Validate chat_id can contain alphanumeric characters, hyphens, and underscores."""
        if not all(c.isalnum() or c in "-_" for c in value):
            logger.warning(f"Invalid chat_id format: {value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid chat_id. Chat ID must be alphanumeric and can include hyphens and underscores.",
            )
        return value

    @field_validator("query")
    def validate_query(cls, value):
        """Validate query is not empty and not profane."""
        if not value.strip():
            logger.warning("Query is empty")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Query cannot be empty."
            )
        elif (
            llm.complete(prompts.profanity_filter.format(query=value)).text.strip()
            == "True"
        ):
            logger.warning(f"Inappropriate content detected in query: {value}")
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail="Sorry, I won't be able to answer your query.",
            )
        return value

    @field_validator("category")
    def validate_category(cls, value):
        """Validate category name is correct."""
        if value not in CategoryConfig.model_fields.keys():
            logger.warning(f"Invalid category: {value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid category. Category must be either {', '.join(CategoryConfig.model_fields.keys())}.",
            )
        return value


class Settings:
    """Application configuration manager loading from YAML."""

    def __init__(self):
        logger.info("Initializing application settings")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.secret = get_secret(self.config)
        logger.info("Application settings initialized successfully")


class PromptManager:
    """Manages loading and accessing prompt templates."""

    @staticmethod
    def load_prompts(config: Dict[str, Any]) -> PromptConfig:
        """Load all prompt templates from the prompts directory."""
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
            return PromptConfig(**prompts)

        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise


class LogManager:
    """CloudWatch logging configuration manager."""

    @staticmethod
    def setup_logging(config: dict, secret: dict) -> None:
        """Initialize CloudWatch logging with instance tracking."""
        try:
            logger = logging.getLogger(__name__)
            logger.info("Setting up CloudWatch logging")

            # Reduce noise from asyncio
            logging.getLogger("asyncio").setLevel(logging.WARNING)

            # Initialize CloudWatch client
            cloudwatch_client = boto3.client(
                "logs",
                aws_access_key_id=secret["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=secret["AWS_SECRET_ACCESS_KEY"],
                region_name=config["AWS_REGION"],
            )

            # Get EC2 instance information
            logger.debug("Retrieving EC2 instance ID")
            ec2_client = boto3.Session().resource(
                "ec2", region_name=config["AWS_REGION"]
            )
            instance_id = ec2_client.meta.client.describe_instances()["Reservations"][
                0
            ]["Instances"][0]["InstanceId"]

            # Configure CloudWatch handler
            cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group=config["CLOUDWATCH_LOG_GROUP"],
                stream_name=instance_id,
                boto3_client=cloudwatch_client,
                use_queues=False,
            )

            # Setup logging configuration
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[cloudwatch_handler],
            )

            # Ensure logs are flushed on shutdown
            atexit.register(lambda: cloudwatch_handler.flush())

            logger.info(f"CloudWatch logging setup complete for instance {instance_id}")

        except Exception as e:
            logger.critical(f"Failed to setup CloudWatch logging: {e}")
            raise


class LLMManager:
    """OpenAI LLM initialization and management."""

    @staticmethod
    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def init_llm(config: Dict[str, Any], secret: Dict[str, Any]) -> OpenAI:
        """Initialize LLM model with retry logic."""
        try:
            model_type = config["LLM_MODEL_TYPE"]
            logger.info(f"Loading LLM model: {config[model_type]['MODEL_NAME']}")

            if model_type == "OPENAI":
                return OpenAI(
                    model=config["OPENAI"]["MODEL_NAME"],
                    api_key=secret["OPENAI_API_KEY"],
                    temperature=config["OPENAI"]["TEMPERATURE"],
                    top_p=config["OPENAI"]["TOP_P"],
                    max_tokens=config["OPENAI"]["MAX_TOKENS"],
                    timeout=config["OPENAI"]["REQUEST_TIMEOUT"],
                )
            elif model_type == "OLLAMA":
                return Ollama(
                    model=config["OLLAMA"]["MODEL_NAME"],
                    temperature=config["OLLAMA"]["TEMPERATURE"],
                    top_p=config["OLLAMA"]["TOP_P"],
                    request_timeout=config["OLLAMA"]["REQUEST_TIMEOUT"],
                    additional_kwargs={
                        "num_ctx": config["OLLAMA"]["CTX_LENGTH"],
                        "num_predict": config["OLLAMA"]["PREDICT_LENGTH"],
                        "cache": False,
                    },
                )
            elif model_type == "ANTHROPIC":
                return Anthropic(
                    model=config["ANTHROPIC"]["MODEL_NAME"],
                    api_key=secret["ANTHROPIC_API_KEY"],
                    temperature=config["ANTHROPIC"]["TEMPERATURE"],
                    max_tokens=config["ANTHROPIC"]["MAX_TOKENS"],
                    timeout=config["ANTHROPIC"]["REQUEST_TIMEOUT"],
                )
            else:
                raise ValueError(f"Invalid LLM model type: {model_type}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise


class AppManager:
    _settings = None
    _llm = None
    _prompts = None
    _app = None

    def __init__(self):
        if AppManager._settings is None:
            AppManager._settings = Settings()
            LogManager.setup_logging(
                AppManager._settings.config, AppManager._settings.secret
            )

    @property
    def settings(self):
        return AppManager._settings

    @property
    def app(self):
        if AppManager._app is None:
            logger.info("Creating FastAPI application")

            AppManager._app = FastAPI(
                title="AltGAN API",
                version="1.0",
                description="API for AltGAN RAG system",
            )

            AppManager._app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        return AppManager._app

    @property
    def prompts(self):
        if AppManager._prompts is None:
            AppManager._prompts = PromptManager.load_prompts(
                AppManager._settings.config
            )

        return AppManager._prompts

    @property
    def llm(self):
        if AppManager._llm is None:
            AppManager._llm = LLMManager.init_llm(
                AppManager._settings.config, AppManager._settings.secret
            )

        return AppManager._llm


app_manager = AppManager()
settings = app_manager.settings
app = app_manager.app
prompts = app_manager.prompts
llm = app_manager.llm


async def save_chat_history(
    chat_id: str, chat_hist: str, config: Dict[str, Any]
) -> None:
    """Background task for chat history summarization and saving to S3."""
    try:
        start_time = time.perf_counter()
        logger.info(f"Starting chat history summarization for chat {chat_id}")

        chat_summ_file = f"{config['S3_CHAT_HISTORY']}/{chat_id}.md"

        # Generate chat summary
        logger.debug("Generating chat summary")
        summarized_hist = llm.complete(
            prompts.history_summarizer.format(chat_history=chat_hist)
        ).text.strip()

        # Save to S3
        logger.debug("Saving summary to S3")
        s3_client = boto3.client("s3")
        s3_client.put_object(
            Bucket=config["S3_LOGS_BUCKET"],
            Key=chat_summ_file,
            Body=summarized_hist,
        )

        duration = time.perf_counter() - start_time
        logger.info(
            f"Chat history summarization completed for chat {chat_id} in {duration:.2f} seconds"
        )

    except ClientError as e:
        logger.error(
            f"S3 error during chat history summarization for chat {chat_id}: {e}"
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during chat history summarization for chat {chat_id}: {e}"
        )
        raise


@app.get("/health", tags=["Health Check"])
async def health_check() -> JSONResponse:
    """Basic health check endpoint."""
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "OK"})


logger.info("Starting API server")
logger.info(f"Server started in {time.perf_counter() - start_time:.2f} seconds")


@app.post("/v1/chat", tags=["Chat API"])
async def get_answer(rag: RAG, background_tasks: BackgroundTasks) -> StreamingResponse:
    """Process chat requests and generate RAG-based responses."""
    try:
        start_time = time.perf_counter()
        logger.info(f"Processing chat request for chat_id: {rag.chat_id}")
        logger.info(
            f"API request started in {time.perf_counter() - start_time:.2f} seconds"
        )

        # Rewrite query
        rag.query = llm.complete(
            prompts.rephrased_query.format(query=rag.query)
        ).text.strip()
        logger.info(f"Updated Query: {rag.query}")

        # Update metadata
        metadata = {}
        if rag.file_name:
            metadata["file_name"] = rag.file_name
        metadata["category"] = rag.category

        # Initialize storage manager and response generator
        logger.debug("Initializing response generator")
        s3_manager = StorageManager(settings.config, settings.secret)

        generate_obj = Generate(
            config=settings.config,
            secret=settings.secret,
            chat_id=rag.chat_id,
            query=rag.query,
            category_name=CategoryConfig.model_fields[rag.category].default,
            metadata=metadata,
            persist_dir=rag.persist_dir,
            collection_name=rag.collection_name,
            s3_manager=s3_manager,
        )

        # Schedule chat history summarization
        logger.debug("Scheduling chat history summarization")
        background_tasks.add_task(
            save_chat_history, rag.chat_id, s3_manager.chat_hist, settings.config
        )

        # Generate streaming response
        logger.debug("Starting response generation")
        response = generate_obj.generate_answer()
        if response is not None:
            duration = time.perf_counter() - start_time
            logger.info(
                f"Chat request processed successfully in {duration:.2f} seconds"
            )
            return StreamingResponse(content=response, media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
        workers=1,
    )
