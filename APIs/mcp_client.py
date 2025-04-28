"""
MCP Client for FastAPI
"""
import requests


class MCPClient:
    def __init__(self, api_url: str):
        """
        Initialize the MCP Client with the URL of the FastAPI or MCP server.
        Example: http://localhost:8000
        """
        self.api_url = api_url.rstrip("/")

    def send_audio(self, audio_path: str) -> dict:
        """
        Sends an audio file to the backend for processing.

        :param audio_path: Path to the local audio file (.wav/.mp3)
        :return: Dictionary with transcript and analysis
        """
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/mpeg")}
            response = requests.post(f"{self.api_url}/upload-audio", files=files)
            response.raise_for_status()
            return response.json()

    def analyze_text(self, session_id: str, _text: str) -> dict:
        """
        Directly sends text to the MCP agent if transcription is already done.

        :param session_id: Identifier for session memory tracking
        :param _text: Journal or spoken content
        :return: Processed output
        """
        payload = {
            "session_id": session_id,
            "user_input": _text
        }
        response = requests.post(f"{self.api_url}/mcp-analyze", json=payload)
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    client = MCPClient(api_url="http://localhost:8000")
    # Example1: Using an audio file
    result = client.send_audio("sample_recording.wav")
    print("Transcript:", result["transcript"])
    print("Analysis:", result["analysis"])

    # Example2: Using text directly
    text = "I'm feeling really anxious about my exams."
    text_response = client.analyze_text(session_id="test123", _text=text)
    print("LLM Response:", text_response)
