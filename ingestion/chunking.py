###############################################################################
# All rights reserved.
###############################################################################

import os
import uuid
from pathlib import Path
from typing import (
    Any,
    Iterator,
    List,
)

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    DocChunk,
    DocMeta,
    HierarchicalChunker,
)
from docling_core.types.doc.document import (
    CodeItem,
    DocItem,
    DoclingDocument,
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel
from pydantic import ConfigDict, PrivateAttr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings

DATA_DIR = Path(settings.data_dir)

enc = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap


class CustomHierarchicalChunker(HierarchicalChunker):
    """
    Custom Hierarchical Chunker that uses a temporary directory for handling
    images.

    This class extends the HierarchicalChunker to allow the use of a temporary
    directory, which is required for saving images from PictureItems.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    temp_dir: str = DATA_DIR  # None se necessario

    def __init__(self, temp_dir: str, *args, **kwargs):
        """
        Initializes a CustomHierarchicalChunker instance.

        Args:
            temp_dir (str): Path to the temporary directory for storing images.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.temp_dir = temp_dir

    def chunk(  # noqa: C901
        self, dl_doc: DoclingDocument, **kwargs: Any
    ) -> Iterator[DocChunk]:
        """
        Chunks the document into smaller pieces based on its structure.

        This method iterates over the document items using the existing method,
        merges list items if applicable, handles section headers and titles,
        processes text items, table items, and picture items. For picture
        items, if the image size exceeds a threshold, it generates an image
        summary.

        Args:
            dl_doc (DoclingDocument): The document to be chunked.
            **kwargs: Additional keyword arguments.

        Yields:
            Iterator[DocChunk]: An iterator of document chunks.
        """
        heading_by_level: dict[int, str] = {}
        list_items: list[DocItem] = []
        # Iterate over document items using the existing method
        for item, level in dl_doc.iterate_items():
            captions = None
            if isinstance(item, DocItem):
                # If merging of list items is enabled, accumulate them
                if self.merge_list_items:
                    if isinstance(item, ListItem) or (
                        isinstance(item, TextItem)
                        and item.label == DocItemLabel.LIST_ITEM
                    ):
                        list_items.append(item)
                        continue
                    elif list_items:
                        yield DocChunk(
                            text=self.delim.join([i.text for i in list_items]),
                            meta=DocMeta(
                                doc_items=list_items,
                                headings=[
                                    heading_by_level[k]
                                    for k in sorted(heading_by_level)
                                ]
                                or None,
                                origin=dl_doc.origin,
                            ),
                        )
                        list_items = []
                # Handle section headers or titles
                if isinstance(item, SectionHeaderItem) or (
                    isinstance(item, TextItem)
                    and item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]
                ):
                    level = (
                        item.level
                        if isinstance(item, SectionHeaderItem)
                        else (0 if item.label == DocItemLabel.TITLE else 1)
                    )
                    heading_by_level[level] = item.text
                    # Remove higher-level headings that have "expired"
                    keys_to_del = [k for k in heading_by_level if k > level]
                    for k in keys_to_del:
                        heading_by_level.pop(k, None)
                    continue
                # Default handling for textual and similar items
                if (
                    isinstance(item, TextItem)
                    or ((not self.merge_list_items) and isinstance(item, ListItem))
                    or isinstance(item, CodeItem)
                ):
                    text = item.text
                elif isinstance(item, TableItem):
                    table_df = item.export_to_dataframe()
                    if table_df.shape[0] < 1 or table_df.shape[1] < 2:
                        continue
                    text = str(table_df.to_markdown())
                    captions = [
                        c.text for c in [r.resolve(dl_doc) for r in item.captions]
                    ] or None
                # Here we include PictureItems – instead of ignoring them, we
                # set a placeholder text
                elif item.label == DocItemLabel.PICTURE:
                    # You can adjust this placeholder to extract figure
                    # information if desired
                    img = item.image.pil_image
                    width, height = img.width, img.height
                    if width * height > 50000:
                        image_name = f"image_{uuid.uuid4()}.png"
                        img_path = os.path.join(self.temp_dir, image_name)

                        with open(img_path, "wb") as f:
                            img.save(f, format="PNG")

                        # Return a base64 image and an image summary; here we
                        # use only the summary
                        text = ""  # imagens ignoradas na PoC (editais sao texto)
                    else:
                        continue
                else:
                    continue

                chunk = DocChunk(
                    text=text,
                    meta=DocMeta(
                        doc_items=[item],
                        headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                        or None,
                        captions=captions,
                        origin=dl_doc.origin,
                    ),
                )
                yield chunk

        if self.merge_list_items and list_items:
            yield DocChunk(
                text=self.delim.join([i.text for i in list_items]),
                meta=DocMeta(
                    doc_items=list_items,
                    headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                    or None,
                    origin=dl_doc.origin,
                ),
            )


class CustomHybridChunker(HybridChunker):
    """
    A hybrid chunker combining hierarchical and semantic segmentation.

    This class processes documents by first splitting them via an internal
    hierarchical chunker and then merging them based on cosine similarity
    (embeddings) and token limits. If a final chunk exceeds the maximum
    token size, a markdown-based subdivision is applied.

    Attributes
    ----------
    temp_dir : str
        Directory path for storing temporary processing artifacts.
    model_config : ConfigDict
        Pydantic model configuration to allow arbitrary types and ignore extra fields.

    Parameters
    ----------
    temp_dir : str
        Path to the directory for temporary files.
    chunk_size : int, default 2000
        Maximum number of tokens allowed per chunk.
    similarity_threshold : float, default 0.6
        Cosine similarity threshold (0 to 1) for merging decision.
    tokenizer_name : str, default "cl100k_base"
        The tiktoken encoding name to be used for token counting.
    **kwargs
        Additional arguments passed to the HybridChunker base class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")
    temp_dir: str
    _custom_max_tokens: int = PrivateAttr()
    _custom_similarity_threshold: float = PrivateAttr()
    _enc_tokenizer: Any = PrivateAttr()
    _embed_model: Any = PrivateAttr()
    _internal_chunker: Any = PrivateAttr()

    def __init__(
        self,
        temp_dir: str,
        chunk_size: int = CHUNK_SIZE,
        similarity_threshold: float = 0.6,
        tokenizer_name: str = "cl100k_base",
        **kwargs,
    ):

        super().__init__(temp_dir=temp_dir, **kwargs)
        self._custom_max_tokens = chunk_size
        self._custom_similarity_threshold = similarity_threshold
        self._enc_tokenizer = tiktoken.get_encoding(tokenizer_name)
        self._embed_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-V2"
        )
        self._internal_chunker = CustomHierarchicalChunker(temp_dir=temp_dir)

    def _count_tokens(self, text: str) -> int:
        """
        Calculates the number of tokens in a given text string.

        Parameters
        ----------
        text : str
            The input string to be tokenized.

        Returns
        -------
        int
            The total token count.
        """
        return len(self._enc_tokenizer.encode(text))

    def _get_embedding(self, text: str):
        """
        Generates a vector embedding for the provided text.

        Parameters
        ----------
        text : str
            The text to be vectorized.

        Returns
        -------
        numpy.ndarray
            The embedding generated by the SentenceTransformer model.
        """
        return self._embed_model.encode([text], convert_to_numpy=True)

    def _is_semantically_similar(self, text1: str, text2: str) -> bool:
        """
        Determines if two text blocks are semantically close.

        Calculates the cosine similarity between the embeddings of both texts
        and compares it against the defined threshold.

        Parameters
        ----------
        text1 : str
            The first text block.
        text2 : str
            The second text block.

        Returns
        -------
        bool
            True if similarity is above or equal to threshold, False otherwise.
        """
        if not text1 or not text2:
            return False
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity >= self._custom_similarity_threshold

    def _merge_chunks_with_semantic_context(
        self, chunks: List[DocChunk]
    ) -> List[DocChunk]:
        """
        Divides a large DocChunk into multiple smaller DocChunks, respecting the Markdown structure.

        Small chunks or chunks with high semantic similarity are merged until
        the token limit (`chunk_size`) is reached.

        Parameters
        ----------
        text : str
            The input string to be tokenized.

        Returns
        -------
        List[DocChunk]
            A list of optimized and merged document chunks.
        """
        if not chunks:
            return []

        merged_list = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            combined_text = self.delim.join([current_chunk.text, next_chunk.text])
            combined_tokens = self._count_tokens(combined_text)

            if combined_tokens <= self._custom_max_tokens:
                if (
                    self._is_semantically_similar(current_chunk.text, next_chunk.text)
                    or len(current_chunk.text) < 300
                ):
                    current_chunk = DocChunk(
                        text=combined_text,
                        meta=DocMeta(
                            doc_items=current_chunk.meta.doc_items
                            + next_chunk.meta.doc_items,
                            headings=current_chunk.meta.headings,
                            origin=current_chunk.meta.origin,
                        ),
                    )
                else:
                    merged_list.append(current_chunk)
                    current_chunk = next_chunk
            else:
                merged_list.append(current_chunk)
                current_chunk = next_chunk

        merged_list.append(current_chunk)
        return merged_list

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[DocChunk]:
        """
        Executes the full chunking pipeline on the document.

        The process follows three stages:
        1. Initial hierarchical decomposition.
        2. Semantic merging based on embeddings and token constraints.
        3. Fallback subdivision via markdown-based chunking for oversized blocks.

        Parameters
        ----------
        dl_doc : DoclingDocument
            The document object to be processed.
        **kwargs : Any
            Additional arguments for the chunking process.

        Yields
        ------
        Iterator[DocChunk]
            A generator of optimized document chunks.
        """
        initial_chunks = list(self._internal_chunker.chunk(dl_doc=dl_doc, **kwargs))
        semantic_chunks = self._merge_chunks_with_semantic_context(initial_chunks)

        for chunk in semantic_chunks:
            if self._count_tokens(chunk.text) > self._custom_max_tokens:
                for sub_chunk in self._apply_markdown_chunking(chunk):
                    yield sub_chunk
            else:
                yield chunk
