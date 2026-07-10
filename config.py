from os import environ
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

@dataclass
class Config:
    croo_api_url: str = field(default_factory=lambda: environ.get("CROO_API_URL", "https://api.croo.network"))
    croo_ws_url: str = field(default_factory=lambda: environ.get("CROO_WS_URL", "wss://api.croo.network/ws"))
    croo_sdk_key: str = field(default_factory=lambda: environ.get("CROO_SDK_KEY", ""))

    google_api_key: str = field(default_factory=lambda: environ.get("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: environ.get("GROQ_API_KEY", ""))
    opencode_api_key: str = field(default_factory=lambda: environ.get("OPENCODE_API_KEY", ""))

    @classmethod
    def from_env(cls) -> "Config":
        return cls()
