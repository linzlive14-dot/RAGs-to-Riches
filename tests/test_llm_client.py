import asyncio
import json

import httpx
import pytest

from app.llm import DailyLimitReached, OpenRouterClient, OpenRouterJudge


def _client(monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]) -> OpenRouterClient:
    queue = iter(responses)
    transport = httpx.MockTransport(lambda request: next(queue))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "app.llm.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("app.llm.asyncio.sleep", no_sleep)
    return OpenRouterClient("test-key", model="test/model:free", max_retries=2)


def _completion(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_per_minute_limit_is_retried_and_wait_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(
        monkeypatch,
        [
            httpx.Response(429, headers={"retry-after": "7"}, text="rate limited"),
            _completion("hello"),
        ],
    )

    assert asyncio.run(client.complete([{"role": "user", "content": "hi"}])) == "hello"
    assert client.requests == 2
    assert client.waited_seconds == 7.0
    assert not client.daily_limit_reached


def test_daily_limit_stops_without_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(
        monkeypatch,
        [httpx.Response(429, text="Rate limit exceeded: free-models-per-day")],
    )

    with pytest.raises(DailyLimitReached):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    assert client.daily_limit_reached
    with pytest.raises(DailyLimitReached):
        asyncio.run(client.complete([{"role": "user", "content": "again"}]))
    assert client.requests == 1


def test_batch_judge_scores_each_answer_and_fails_missing_ones() -> None:
    class FakeClient:
        async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
            items = json.loads(messages[1]["content"])["items"]
            assert [item["id"] for item in items] == ["a", "b", "c"]
            return json.dumps(
                {
                    "results": [
                        {"id": "a", "grounded": True, "citations_accurate": True},
                        {"id": "b", "grounded": True, "citations_accurate": False},
                    ]
                }
            )

    items = [
        {"id": key, "answer": "text", "evidence": {}, "rubric": {}} for key in ("a", "b", "c")
    ]
    verdicts = asyncio.run(OpenRouterJudge(FakeClient()).score_batch(items))

    assert verdicts["a"] == {"grounded": True, "citations_accurate": True}
    assert verdicts["b"] == {"grounded": True, "citations_accurate": False}
    assert verdicts["c"] == {"grounded": False, "citations_accurate": False}
