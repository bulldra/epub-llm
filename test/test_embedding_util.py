"""Tests for embedding_util module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embedding_util import (
    ModelPair,
    build_faiss_index,
    create_context_from_query,
    create_embeddings_from_texts,
    embed_texts_and_save,
    load_and_search,
    load_embeddings,
    save_embeddings,
    search_similar,
)


class TestModelPair:  # pylint: disable=too-few-public-methods
    """Test cases for ModelPair dataclass."""

    def test_model_pair_creation(self):
        """Test ModelPair creation with model and tokenizer."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        model_pair = ModelPair(model=mock_model, tokenizer=mock_tokenizer)

        assert model_pair.model is mock_model
        assert model_pair.tokenizer is mock_tokenizer


class TestEmbeddingUtil:
    """Test cases for embedding_util functions."""

    def test_create_embeddings_from_texts(self):
        """Test creating embeddings from texts."""
        texts = ["Hello world", "This is a test"]
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer output
        mock_tokenizer.batch_encode_plus.return_value = {
            "input_ids": "mock_ids",
            "attention_mask": "mock_mask",
        }

        # Mock model output - need to return separate arrays for each batch
        mock_output1 = MagicMock()
        mock_output1.text_embeds = np.array([[1.0, 2.0, 3.0]])
        mock_output2 = MagicMock()
        mock_output2.text_embeds = np.array([[4.0, 5.0, 6.0]])
        mock_model.side_effect = [mock_output1, mock_output2]

        result = create_embeddings_from_texts(
            texts, mock_model, mock_tokenizer, batch_size=1
        )

        assert result.shape == (2, 3)
        assert np.array_equal(result, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
            texts = ["text1", "text2"]
            base_path = os.path.join(temp_dir, "test_embeddings")

            save_embeddings(embeddings, texts, base_path)

            # Check files were created
            assert os.path.exists(base_path + ".npy")
            assert os.path.exists(base_path + ".json")

            # Load and verify
            loaded_embeddings, loaded_texts = load_embeddings(base_path)

            assert np.array_equal(loaded_embeddings, embeddings)
            assert loaded_texts == texts

    def test_build_faiss_index_2d(self):
        """Test building FAISS index with 2D embeddings."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        index = build_faiss_index(embeddings)

        assert index.ntotal == 2
        assert index.d == 2

    def test_build_faiss_index_1d(self):
        """Test building FAISS index with 1D embeddings (auto-reshape)."""
        embeddings = np.array([1.0, 2.0], dtype=np.float32)

        index = build_faiss_index(embeddings)

        assert index.ntotal == 1
        assert index.d == 2

    def test_build_faiss_index_invalid_shape(self):
        """Test building FAISS index with invalid shape raises error."""
        embeddings = np.array([[[1.0, 2.0]]], dtype=np.float32)

        with pytest.raises(ValueError, match="embeddings must be 2D"):
            build_faiss_index(embeddings)

    def test_search_similar(self):
        """Test searching for similar texts."""
        # Setup mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        model_pair = ModelPair(model=mock_model, tokenizer=mock_tokenizer)

        # Mock tokenizer output
        mock_tokenizer.batch_encode_plus.return_value = {
            "input_ids": "mock_ids",
            "attention_mask": "mock_mask",
        }

        # Mock model output for query
        mock_output = MagicMock()
        mock_output.text_embeds = np.array([[1.0, 2.0]], dtype=np.float32)
        mock_model.return_value = mock_output

        # Setup test data
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        texts = ["text1", "text2", "text3"]
        index = build_faiss_index(embeddings)

        result = search_similar("query", model_pair, index, texts, top_k=2)

        assert len(result) == 2
        assert all(isinstance(r, tuple) and len(r) == 3 for r in result)
        assert all(isinstance(r[0], int | np.integer) for r in result)  # index
        assert all(isinstance(r[1], float | np.floating) for r in result)  # distance
        assert all(isinstance(r[2], str) for r in result)  # text

    def test_create_context_from_query(self):
        """Test creating context from query results."""
        # Setup mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        model_pair = ModelPair(model=mock_model, tokenizer=mock_tokenizer)

        # Mock the search_similar function
        with patch("src.embedding_util.search_similar") as mock_search:
            mock_search.return_value = [
                (0, 0.1, "First result"),
                (1, 0.2, "Second result"),
            ]

            embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            texts = ["text1", "text2"]
            index = build_faiss_index(embeddings)

            result = create_context_from_query(
                "query", model_pair, index, texts, top_k=2
            )

            assert result == "First result\n\n------\n\nSecond result"

    def test_embed_texts_and_save(self):
        """Test embedding texts and saving them."""
        with tempfile.TemporaryDirectory() as temp_dir:
            texts = ["text1", "text2"]
            out_path = os.path.join(temp_dir, "test")
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            with (
                patch("src.embedding_util.create_embeddings_from_texts") as mock_create,
                patch("src.embedding_util.save_embeddings") as mock_save,
            ):
                mock_embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
                mock_create.return_value = mock_embeddings

                embed_texts_and_save(texts, out_path, mock_model, mock_tokenizer)

                mock_create.assert_called_once_with(
                    texts=texts,
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    batch_size=32,
                )
                mock_save.assert_called_once_with(mock_embeddings, texts, out_path)

    def test_load_and_search(self):
        """Test loading embeddings and searching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(temp_dir, "test")
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            model_pair = ModelPair(model=mock_model, tokenizer=mock_tokenizer)

            # Create test files
            embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            texts = ["text1", "text2"]
            save_embeddings(embeddings, texts, base_path)

            with patch("src.embedding_util.search_similar") as mock_search:
                mock_search.return_value = [(0, 0.1, "text1")]

                result = load_and_search("query", base_path, model_pair, top_k=1)

                assert result == [(0, 0.1, "text1")]
                mock_search.assert_called_once()

    def test_load_embeddings_file_not_found(self):
        """Test loading embeddings when files don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(temp_dir, "nonexistent")

            with pytest.raises(FileNotFoundError):
                load_embeddings(base_path)

    def test_save_embeddings_creates_files(self):
        """Test that save_embeddings creates the expected files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = np.array([[1.0, 2.0]])
            texts = ["test"]
            base_path = os.path.join(temp_dir, "test")

            save_embeddings(embeddings, texts, base_path)

            # Check numpy file
            loaded_embeddings = np.load(base_path + ".npy")
            assert np.array_equal(loaded_embeddings, embeddings)

            # Check JSON file
            with open(base_path + ".json", encoding="utf-8") as f:
                loaded_texts = json.load(f)
            assert loaded_texts == texts
