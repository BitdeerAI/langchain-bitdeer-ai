"""Standard LangChain interface tests."""

from langchain_core.embeddings import Embeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_bitdeer_ai import BitdeerAIEmbeddings

MODEL_NAME = "BAAI/bge-m3"


class TestBitdeerAIEmbeddingsStandard(EmbeddingsIntegrationTests):
    """Run LangChain's standard embeddings integration tests."""

    @property
    def embeddings_class(self) -> type[Embeddings]:
        """Embeddings class under test."""
        return BitdeerAIEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        """Parameters for constructing the embeddings."""
        return {"model": MODEL_NAME}
