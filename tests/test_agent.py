import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os
from dataclasses import dataclass

# Add the project root to sys.path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the livekit agents imports since they might not be available in test env
sys.modules['livekit.agents'] = MagicMock()
sys.modules['livekit.agents.voice'] = MagicMock()
sys.modules['livekit.agents.types'] = MagicMock()
sys.modules['livekit.plugins'] = MagicMock()
sys.modules['livekit.plugins.noise_cancellation'] = MagicMock()
sys.modules['livekit.plugins.silero'] = MagicMock()

from src.agent import (
    VoiceAgent,
    extract_caller_number_from_room,
    extract_transcript_from_chat_ctx,
    SpamDetectionConfig,
)


class MockChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockChatContext:
    def __init__(self, messages):
        self.messages = messages


class MockParticipant:
    def __init__(self, attributes):
        self.attributes = attributes


class MockRoom:
    def __init__(self, participants):
        self.remote_participants = participants


class TestAgentFunctionality:
    """Test cases for agent.py core functionality"""
    
    def test_extract_transcript_empty(self):
        """Test transcript extraction with empty chat context"""
        ctx = MockChatContext([])
        transcript = extract_transcript_from_chat_ctx(ctx)
        assert transcript == ""
    
    def test_extract_transcript_caller_only(self):
        """Test transcript extraction with caller messages only"""
        messages = [
            MockChatMessage("user", "Hello, is this the IRS?"),
            MockChatMessage("user", "I'm calling about my tax refund."),
        ]
        ctx = MockChatContext(messages)
        transcript = extract_transcript_from_chat_ctx(ctx)
        
        expected = "Caller: Hello, is this the IRS?\nCaller: I'm calling about my tax refund."
        assert transcript == expected
    
    def test_extract_transcript_agent_only(self):
        """Test transcript extraction with agent messages only"""
        messages = [
            MockChatMessage("assistant", "Hello, how can I help you?"),
            MockChatMessage("assistant", "Please tell me more."),
        ]
        ctx = MockChatContext(messages)
        transcript = extract_transcript_from_chat_ctx(ctx)
        
        expected = "Agent: Hello, how can I help you?\nAgent: Please tell me more."
        assert transcript == expected
    
    def test_extract_transcript_mixed(self):
        """Test transcript extraction with mixed caller/agent messages"""
        messages = [
            MockChatMessage("assistant", "Hello, this line is open."),
            MockChatMessage("user", "Hi, I'm calling about your car warranty."),
            MockChatMessage("assistant", "I see, go on."),
            MockChatMessage("user", "It's about to expire and I want to extend it."),
            MockChatMessage("assistant", "Okay, thank you for calling."),
        ]
        ctx = MockChatContext(messages)
        transcript = extract_transcript_from_chat_ctx(ctx)
        
        expected = (
            "Agent: Hello, this line is open.\n"
            "Caller: Hi, I'm calling about your car warranty.\n"
            "Agent: I see, go on.\n"
            "Caller: It's about to expire and I want to extend it.\n"
            "Agent: Okay, thank you for calling."
        )
        assert transcript == expected
    
    def test_extract_transcript_filters_empty_messages(self):
        """Test that empty or whitespace-only messages are filtered out"""
        messages = [
            MockChatMessage("assistant", "Hello"),
            MockChatMessage("user", ""),
            MockChatMessage("user", "   "),
            MockChatMessage("user", "\n\t"),
            MockChatMessage("assistant", "How can I help?"),
        ]
        ctx = MockChatContext(messages)
        transcript = extract_transcript_from_chat_ctx(ctx)
        
        expected = "Agent: Hello\nAgent: How can I help?"
        assert transcript == expected

    def test_extract_caller_number_from_room(self):
        """Test caller number extraction from LiveKit SIP attributes"""
        room = MockRoom(
            {
                "caller": MockParticipant({"sip.phoneNumber": "+919876543210"}),
            }
        )

        assert extract_caller_number_from_room(room) == "+919876543210"

    def test_extract_caller_number_from_room_missing(self):
        """Test caller number extraction when no SIP number is present"""
        room = MockRoom({"caller": MockParticipant({"sip.callID": "abc"})})

        assert extract_caller_number_from_room(room) is None
    
    def test_spam_detection_config_defaults(self):
        """Test SpamDetectionConfig default values"""
        config = SpamDetectionConfig()
        assert config.call_duration_seconds == 20.0
        assert config.classification_model == "gpt-4o-mini"
    
    def test_spam_detection_config_custom(self):
        """Test SpamDetectionConfig with custom values"""
        config = SpamDetectionConfig(call_duration_seconds=30.0, classification_model="gpt-3.5-turbo")
        assert config.call_duration_seconds == 30.0
        assert config.classification_model == "gpt-3.5-turbo"
    
    def test_voice_agent_initialization(self):
        """Test that VoiceAgent initializes correctly"""
        # This test checks that we can instantiate the VoiceAgent
        # without errors (mocking the complex dependencies)
        with patch('src.agent.load_config'), patch('src.agent.load_instructions'):
            with patch('src.agent.create_stt'), patch('src.agent.create_llm'), patch('src.agent.create_tts'):
                agent = VoiceAgent()
                assert agent is not None
                assert hasattr(agent, '_config')
                assert hasattr(agent, '_call_ended_event')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
