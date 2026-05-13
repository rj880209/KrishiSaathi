"""
Voice processing module for multilingual support
Supports: Hindi, Telugu, Marathi, Bengali, Tamil
Uses Groq Speech-to-Text with fallback options
"""

import os
import io
import wave
from typing import Optional, Dict
from enum import Enum


class Language(Enum):
    """Supported languages"""
    HINDI = "hi"
    TELUGU = "te"
    MARATHI = "mr"
    BENGALI = "bn"
    TAMIL = "ta"
    ENGLISH = "en"


# Language display names
LANGUAGE_NAMES = {
    Language.HINDI: "हिंदी (Hindi)",
    Language.TELUGU: "తెలుగు (Telugu)",
    Language.MARATHI: "मराठी (Marathi)",
    Language.BENGALI: "বাংলা (Bengali)",
    Language.TAMIL: "தமிழ் (Tamil)",
    Language.ENGLISH: "English"
}


class VoiceProcessor:
    """Handle speech-to-text conversion using Groq"""
    
    def __init__(self, groq_client=None):
        self.client = groq_client
        self.supported_languages = [lang.value for lang in Language]
    
    def transcribe_audio(self, audio_file_path: str, language: str = "hi") -> Optional[str]:
        """
        Transcribe audio file to text using Groq
        
        Args:
            audio_file_path: Path to the audio file (WAV format)
            language: Language code (hi, te, mr, bn, ta, en)
        
        Returns:
            Transcribed text or None if failed
        """
        try:
            if not self.client:
                return self._fallback_transcription(audio_file_path, language)
            
            # Read audio file
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
            
            # Use Groq's transcription API
            transcription = self.client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), audio_data),
                model="whisper-large-v3",
                language=language,
                response_format="text"
            )
            
            return transcription.text.strip()
            
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            return self._fallback_transcription(audio_file_path, language)
    
    def _fallback_transcription(self, audio_file_path: str, language: str) -> Optional[str]:
        """Fallback transcription method when Groq is unavailable"""
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_file_path) as source:
                audio = recognizer.record(source)
            
            # Try Google Speech Recognition (free, no API key needed for limited use)
            text = recognizer.recognize_google(audio, language=language)
            return text
            
        except Exception as e:
            print(f"Fallback transcription failed: {str(e)}")
            return None
    
    def text_to_speech(self, text: str, language: str = "hi") -> Optional[bytes]:
        """
        Convert text to speech (placeholder for future TTS integration)
        
        Args:
            text: Text to convert to speech
            language: Language code
        
        Returns:
            Audio bytes or None
        """
        # TODO: Integrate with Groq TTS or other TTS service
        # For now, return None as this is a placeholder
        print("TTS not yet implemented")
        return None
    
    def validate_audio_file(self, file_path: str) -> bool:
        """Validate audio file format"""
        try:
            if not os.path.exists(file_path):
                return False
            
            # Check if it's a valid WAV file
            with wave.open(file_path, 'rb') as wf:
                # Check basic properties
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                
                # Accept mono or stereo, 16-bit, reasonable sample rate
                if channels not in [1, 2]:
                    return False
                if sample_width != 2:
                    return False
                if framerate < 8000 or framerate > 48000:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_language_options(self) -> Dict[str, str]:
        """Get available language options for UI"""
        return {lang.value: name for lang, name in LANGUAGE_NAMES.items()}


def create_sample_audio(duration_seconds: int = 3, filename: str = "sample.wav"):
    """Create a sample silent audio file for testing"""
    import wave
    import struct
    
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        
        # Generate silence
        for _ in range(num_samples):
            wf.writeframes(struct.pack('h', 0))
    
    return filename


if __name__ == "__main__":
    # Test voice processor
    processor = VoiceProcessor()
    print("Supported languages:", processor.get_language_options())
    
    # Create sample audio for testing
    sample_file = create_sample_audio(2, "test_sample.wav")
    print(f"Created sample audio: {sample_file}")
    print(f"Valid audio file: {processor.validate_audio_file(sample_file)}")
