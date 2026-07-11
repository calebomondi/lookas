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

    base_price: float = field(default_factory=lambda: float(environ.get("BASE_PRICE", "0.05")))
    per_analyst_price: float = field(default_factory=lambda: float(environ.get("PER_ANALYST_PRICE", "0.08")))
    per_word_price: float = field(default_factory=lambda: float(environ.get("PER_WORD_PRICE", "0.05")))

    @classmethod
    def from_env(cls) -> "Config":
        return cls()
