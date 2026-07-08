###############################################################################
# All rights reserved.
###############################################################################

import time
from logging import getLogger

import voyageai
from langchain_voyageai import VoyageAIEmbeddings

from config.settings import settings

sleep_time = 0.5
VOYAGE_MODEL = settings.voyage_model
ANTHROPIC_MODEL = settings.anthropic_model
VOYAGE_API_KEY = settings.voyage_api_key

batch_valor = 100

logger = getLogger(__name__)


class VoyageEmbedding:
    def __init__(self, collection_name: str, embedding_model=VOYAGE_MODEL):
        """
        Initialize the VoyageEmbedding instance with Voyage AI connection.

        Parameters
        ----------
        collection_name : str
            The name of the collection to be used for embeddings.
        embedding_model : str, optional
            The embedding model to use. Defaults to VOYAGE_MODEL environment variable.

        Returns
        -------
        None

        Notes
        -----
        Initializes the Voyage AI client and sets up the embedding function.
        """
        self.client = self.get_voyage_client()
        self.collection_name = collection_name
        self.embedding_function = VoyageAIEmbeddings(model=embedding_model)
        self.embedding_model = embedding_model

    def close_voyage(self):
        """
        Close the connection with Voyage AI client.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If an error occurs while disconnecting the Voyage AI client.

        Notes
        -----
        Logs a warning if attempting to close a non-existent client.
        """
        try:
            if self.client:
                self.client = None
                logger.success("Client successfully disconnected")
            else:
                logger.warning("No client to disconnect.")
        except Exception as e:
            del self.client
            raise RuntimeError("Error disconnecting Voyage client") from e

    def get_voyage_client(self) -> voyageai.Client:
        """
        Create and return a Voyage AI client instance.

        Parameters
        ----------
        None

        Returns
        -------
        voyageai.Client
            An authenticated Voyage AI client instance.

        Raises
        ------
        RuntimeError
            If the VOYAGE_API_KEY environment variable is not set or if the
            connection to Voyage AI fails.

        Notes
        -----
        Requires the VOYAGE_API_KEY environment variable to be properly configured.
        """
        try:
            if not VOYAGE_API_KEY:
                logger.error("Error locating Voyage API key")
                raise RuntimeError("Variavel de sistema faltando preencha a env")

            client = voyageai.Client(api_key=VOYAGE_API_KEY)

            return client
        except Exception as e:
            raise RuntimeError("Erro ao criar a conexão com a Voyage") from e

    def Vectorize_documents(
        self, chunks: list[str], batch_size: int = batch_valor
    ) -> list[list[float]]:
        """
        Vectorize a list of text chunks into embeddings using Voyage AI.

        Parameters
        ----------
        chunks : list[str]
            A list of text strings to be vectorized into embeddings.
        batch_size : int, optional
            The number of chunks to process in each batch. Defaults to batch_valor (100).

        Returns
        -------
        list[list[float]]
            A list of embedding vectors, where each vector is a list of floats.
            Returns an empty list if the input chunks list is empty.

        Raises
        ------
        Exception
            If a critical error occurs during the vectorization process.

        Notes
        -----
        - Processes chunks in batches to optimize API calls and memory usage.
        - Includes a sleep interval between batches to avoid rate limiting.
        - Logs warnings if the chunks list is empty or missing.
        - Logs debug information for each completed batch.
        """
        try:
            logger.info(
                f"Inicializando a vetorização de {len(chunks)} chunks. Modelo aplicado {self.embedding_model}"
            )
            all_embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                batch_embeddings = self.embedding_function.embed_documents(batch)

                all_embeddings.extend(batch_embeddings)
                logger.debug(
                    f"Vetorização concluida {i // batch_size + 1} processado. Vetores gerados {len(batch_embeddings)}"
                )
                time.sleep(sleep_time)

            return all_embeddings

        except Exception as e:
            logger.exception("Critical error vectorizing documents.")
            raise e
