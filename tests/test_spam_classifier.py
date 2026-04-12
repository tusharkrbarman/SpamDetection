import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import json
import sys
import os

# Add the project root to sys.path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spam_classifier import classify_transcript, ClassificationResult


class TestSpamClassifier:
    """Test cases for spam_classifier.py"""
    
    @pytest.mark.asyncio
    async def test_empty_transcript(self):
        """Test classification with empty transcript"""
        result = await classify_transcript("")
        
        assert isinstance(result, ClassificationResult)
        assert result.is_spam == False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"
        assert result.evidence_lines == []
        assert result.full_transcript == ""
    
    @pytest.mark.asyncio
    async def test_whitespace_transcript(self):
        """Test classification with whitespace-only transcript"""
        result = await classify_transcript("   \n\t  ")
        
        assert isinstance(result, ClassificationResult)
        assert result.is_spam == False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"
        assert result.evidence_lines == []
        assert result.full_transcript == "   \n\t  "
    
    @pytest.mark.asyncio
    async def test_spam_classification_mock(self):
        """Test spam classification with mocked OpenAI response"""
        # Mock transcript that should be classified as spam
        transcript = "Caller: Hello, I'm calling from Bank of America about your credit card offer..."
        
        # Mock the OpenAI client and response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_spam": True,
            "confidence": 0.95,
            "reason": "Unsolicited financial product offer",
            "evidence_lines": ["Caller: Hello, I'm calling from Bank of America about your credit card offer..."]
        })
        
        with patch('src.spam_classifier.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_client
            
            # Set required environment variable
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OLLAMA_API_KEY": ""}):
                result = await classify_transcript(transcript)
        
        assert isinstance(result, ClassificationResult)
        assert result.is_spam == True
        assert result.confidence == 0.95
        assert result.reason == "Unsolicited financial product offer"
        assert len(result.evidence_lines) == 1
        assert "Bank of America" in result.evidence_lines[0]
        assert result.full_transcript == transcript
    
    @pytest.mark.asyncio
    async def test_legitimate_classification_mock(self):
        """Test legitimate classification with mocked OpenAI response"""
        # Mock transcript that should be classified as legitimate
        transcript = "Caller: Hi, this is John from Nextdoor. I found your lost dog and wanted to return it."
        
        # Mock the OpenAI client and response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_spam": False,
            "confidence": 0.90,
            "reason": "Legitimate community member returning lost pet",
            "evidence_lines": []
        })
        
        with patch('src.spam_classifier.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_client
            
            # Set required environment variable
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = await classify_transcript(transcript)
        
        assert isinstance(result, ClassificationResult)
        assert result.is_spam == False
        assert result.confidence == 0.90
        assert result.reason == "Legitimate community member returning lost pet"
        assert result.evidence_lines == []
        assert result.full_transcript == transcript
    
    @pytest.mark.asyncio
    async def test_classification_error_handling(self):
        """Test that classification handles OpenAI API errors gracefully"""
        transcript = "Some test transcript"
        
        # Mock OpenAI to raise an exception
        mock_response_error = MagicMock()
        mock_response_error.choices = [MagicMock()]
        mock_response_error.choices[0].message.content = json.dumps({
            "is_spam": False,
            "confidence": 0.0,
            "reason": "Error",
            "evidence_lines": []
        })
        
        with patch('src.spam_classifier.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
            mock_openai_class.return_value = mock_client
            
            # Set required environment variable
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OLLAMA_API_KEY": ""}):
                result = await classify_transcript(transcript)
        
        # Should return a safe fallback result
        assert isinstance(result, ClassificationResult)
        assert result.is_spam == False
        assert result.confidence == 0.0
        assert "Classification error:" in result.reason
        assert result.evidence_lines == []
        assert result.full_transcript == transcript


if __name__ == "__main__":
    pytest.main([__file__, "-v"])