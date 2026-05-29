from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping
from urllib import error, parse, request


@dataclass(frozen=True)
class LLMSpec:
    name: str
    provider: str
    model: str
    api_key_env: str
    api_base: str | None = None
    timeout_sec: int = 90
    max_retries: int = 2
    max_output_tokens: int = 700
    temperature: float = 0.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LLMSpec":
        return cls(
            name=str(payload.get("name") or ""),
            provider=str(payload.get("provider") or ""),
            model=str(payload.get("model") or ""),
            api_key_env=str(payload.get("api_key_env") or ""),
            api_base=None if payload.get("api_base") in {None, ""} else str(payload.get("api_base")),
            timeout_sec=int(payload.get("timeout_sec", 90)),
            max_retries=int(payload.get("max_retries", 2)),
            max_output_tokens=int(payload.get("max_output_tokens", 700)),
            temperature=float(payload.get("temperature", 0.0)),
            extra_headers={str(k): str(v) for k, v in dict(payload.get("extra_headers") or {}).items()},
            extra_body=dict(payload.get("extra_body") or {}),
        )


class LLMRequestError(RuntimeError):
    pass


def _is_timeout_error(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, error.URLError):
        return _is_timeout_error(exc.reason)
    text = str(exc).strip().lower()
    return "timed out" in text or "timeout" in text


def _join_openai_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and str(item.get("type")) == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts).strip()
    return str(content or "").strip()


def _join_parts(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return str(parts or "").strip()
    out: list[str] = []
    for part in parts:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict):
            if "text" in part:
                out.append(str(part.get("text") or ""))
    return "".join(out).strip()


def _json_post(url: str, body: Mapping[str, Any], headers: Mapping[str, str], timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(dict(body)).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        req.add_header(str(key), str(value))
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise LLMRequestError(f"HTTP {exc.code} from {url}: {payload}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise LLMRequestError(f"Request to {url} timed out after {timeout_sec}s: {exc}") from exc
    except error.URLError as exc:
        if _is_timeout_error(exc.reason):
            raise LLMRequestError(f"Request to {url} timed out after {timeout_sec}s: {exc.reason}") from exc
        raise LLMRequestError(f"Request to {url} failed: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise LLMRequestError(f"Non-JSON response from {url}: {raw[:500]}") from exc
    if not isinstance(parsed, dict):
        raise LLMRequestError(f"Unexpected response payload from {url}: {type(parsed)!r}")
    return parsed


class BaseLLMClient:
    def __init__(self, spec: LLMSpec) -> None:
        self.spec = spec
        api_key = os.environ.get(spec.api_key_env, "").strip()
        if not api_key:
            raise LLMRequestError(
                f"Environment variable {spec.api_key_env} is not set for LLM '{spec.name}'."
            )
        self.api_key = api_key

    def generate(self, prompt_text: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.spec.max_retries + 1):
            timeout_sec = self._timeout_for_attempt(attempt)
            try:
                return self._generate_once(prompt_text, timeout_sec)
            except Exception as exc:
                last_error = exc
                if attempt >= self.spec.max_retries:
                    break
                time.sleep(self._retry_delay_sec(attempt, exc))
        raise LLMRequestError(f"LLM '{self.spec.name}' request failed: {last_error}") from last_error

    def _timeout_for_attempt(self, attempt: int) -> int:
        timeout_sec = max(int(self.spec.timeout_sec), 1)
        if attempt <= 0:
            return timeout_sec
        timeout_cap = max(timeout_sec, 300)
        grown_timeout = max(timeout_sec + (60 * attempt), int(round(timeout_sec * (2.0**attempt))))
        return min(grown_timeout, timeout_cap)

    def _retry_delay_sec(self, attempt: int, exc: Exception) -> float:
        if _is_timeout_error(exc):
            return min(5.0 * (attempt + 1), 20.0)
        return min(2.0 * (attempt + 1), 5.0)

    def _generate_once(self, prompt_text: str, timeout_sec: int) -> str:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    @staticmethod
    def _supports_gpt5_token_param(model_name: str) -> bool:
        return str(model_name or "").strip().lower().startswith("gpt-5")

    @staticmethod
    def _supports_temperature_param(model_name: str) -> bool:
        # OpenAI reasoning models (for example o3/o4) do not support overriding temperature.
        return not str(model_name or "").strip().lower().startswith(("o1", "o3", "o4"))

    @staticmethod
    def _should_retry_with_alt_token_param(error_text: str, param_name: str) -> bool:
        text = str(error_text or "").lower()
        return param_name.lower() in text and (
            "unsupported parameter" in text or "unknown parameter" in text or "not supported" in text
        )

    @staticmethod
    def _token_budget_candidates(base_budget: int, *, cap: int = 4096) -> list[int]:
        budget = max(int(base_budget), 1)
        candidates = [budget]
        for next_budget in (
            max(budget * 2, budget + 512),
            max(budget * 3, budget + 1024),
        ):
            capped = min(int(next_budget), int(cap))
            if capped > candidates[-1]:
                candidates.append(capped)
        return candidates

    @staticmethod
    def _finish_reason(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("finish_reason") or "")

    def _generate_once(self, prompt_text: str, timeout_sec: int) -> str:
        base = self.spec.api_base or "https://api.openai.com/v1/chat/completions"
        url = base if base.rstrip("/").endswith("/chat/completions") else base.rstrip("/") + "/chat/completions"
        base_body: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [{"role": "user", "content": prompt_text}],
            **self.spec.extra_body,
        }
        if self._supports_temperature_param(self.spec.model):
            base_body["temperature"] = self.spec.temperature

        has_token_limit = "max_tokens" in base_body or "max_completion_tokens" in base_body
        preferred_param = "max_completion_tokens" if self._supports_gpt5_token_param(self.spec.model) else "max_tokens"
        alternate_param = "max_tokens" if preferred_param == "max_completion_tokens" else "max_completion_tokens"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            **self.spec.extra_headers,
        }

        param_candidates = [preferred_param] if has_token_limit else [preferred_param, alternate_param]
        budget_candidates = [self.spec.max_output_tokens] if has_token_limit else self._token_budget_candidates(self.spec.max_output_tokens)
        last_empty_payload: Mapping[str, Any] | None = None
        last_error: Exception | None = None

        for param_name in param_candidates:
            unsupported_param = False
            for budget in budget_candidates:
                body = dict(base_body)
                if not has_token_limit:
                    body[param_name] = int(budget)
                try:
                    payload = _json_post(url, body, headers, timeout_sec)
                except LLMRequestError as exc:
                    last_error = exc
                    if not has_token_limit and self._should_retry_with_alt_token_param(str(exc), param_name):
                        unsupported_param = True
                        break
                    raise

                choices = payload.get("choices") or []
                if not choices:
                    raise LLMRequestError(f"OpenAI response contained no choices: {payload}")
                message = choices[0].get("message") or {}
                content = _join_openai_content(message.get("content"))
                if content:
                    return content

                last_empty_payload = payload
                if self._finish_reason(payload) == "length" and (not has_token_limit) and budget < budget_candidates[-1]:
                    continue
                raise LLMRequestError(f"OpenAI response contained empty content: {payload}")

            if unsupported_param:
                continue

        if last_empty_payload is not None:
            raise LLMRequestError(f"OpenAI response contained empty content: {last_empty_payload}")
        if last_error is not None:
            raise last_error
        raise LLMRequestError("OpenAI request failed without a usable response payload.")


class AnthropicClient(BaseLLMClient):
    def _generate_once(self, prompt_text: str, timeout_sec: int) -> str:
        base = self.spec.api_base or "https://api.anthropic.com/v1/messages"
        url = base if base.rstrip("/").endswith("/messages") else base.rstrip("/") + "/messages"
        body: dict[str, Any] = {
            "model": self.spec.model,
            "max_tokens": self.spec.max_output_tokens,
            "temperature": self.spec.temperature,
            "messages": [{"role": "user", "content": prompt_text}],
            **self.spec.extra_body,
        }
        payload = _json_post(
            url,
            body,
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                **self.spec.extra_headers,
            },
            timeout_sec,
        )
        content = _join_parts(payload.get("content"))
        if not content:
            raise LLMRequestError(f"Anthropic response contained empty content: {payload}")
        return content


class GoogleClient(BaseLLMClient):
    def _generate_once(self, prompt_text: str, timeout_sec: int) -> str:
        base = self.spec.api_base or "https://generativelanguage.googleapis.com/v1beta"
        if ":generateContent" in base:
            url = base
            if "key=" not in url:
                joiner = "&" if "?" in url else "?"
                url = f"{url}{joiner}key={parse.quote(self.api_key)}"
        else:
            model = parse.quote(self.spec.model, safe="")
            url = f"{base.rstrip('/')}/models/{model}:generateContent?key={parse.quote(self.api_key)}"
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": self.spec.temperature,
                "maxOutputTokens": self.spec.max_output_tokens,
            },
            **self.spec.extra_body,
        }
        payload = _json_post(url, body, self.spec.extra_headers, timeout_sec)
        candidates = payload.get("candidates") or []
        if not candidates:
            raise LLMRequestError(f"Google response contained no candidates: {payload}")
        content = _join_parts(((candidates[0].get("content") or {}).get("parts")))
        if not content:
            raise LLMRequestError(f"Google response contained empty content: {payload}")
        return content


def build_llm_client(spec: LLMSpec) -> BaseLLMClient:
    provider = str(spec.provider or "").strip().lower()
    if provider in {"openai", "chatgpt"}:
        return OpenAIClient(spec)
    if provider in {"anthropic", "claude"}:
        return AnthropicClient(spec)
    if provider in {"google", "gemini"}:
        return GoogleClient(spec)
    raise LLMRequestError(f"Unsupported LLM provider '{spec.provider}' for {spec.name}.")
