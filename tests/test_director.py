from agent.director import (
    Director,
    _claim_focus_guidance,
    _looks_like_evaluative_meta_claim,
)
from agent.models import Claim, ResearchResult, VerdictLabel


def test_detects_evaluative_meta_claim_headlines():
    assert _looks_like_evaluative_meta_claim(
        "Elon Musk Amplifies Baseless Claim About COVID-19 Vaccine"
    )
    assert _looks_like_evaluative_meta_claim(
        "Article debunked misleading claim about election fraud"
    )
    assert not _looks_like_evaluative_meta_claim("Vaccines cause autism.")


def test_claim_focus_guidance_adds_meta_claim_instructions():
    guidance = _claim_focus_guidance(
        "Elon Musk Amplifies Baseless Claim About COVID-19 Vaccine"
    )

    assert "This claim is an evaluative headline about another claim." in guidance
    assert "Do NOT give a verdict solely for the embedded claim" in guidance


def test_synthesize_verdict_includes_exact_claim_guidance():
    captured: dict[str, str] = {}

    def fake_chat(prompt: str):
        captured["prompt"] = prompt
        return {
            "verdict": "TRUE",
            "confidence": 0.8,
            "summary": "The headline-level claim is supported by the evidence.",
            "key_evidence": ["Evidence bullet."],
        }

    director = object.__new__(Director)
    director._chat = fake_chat

    claim = Claim(
        id="claim-1",
        text="Elon Musk Amplifies Baseless Claim About COVID-19 Vaccine",
        source="FactCheck.org",
    )
    research = ResearchResult(
        claim_id=claim.id,
        findings="Research findings here.",
        sources=[],
        fetched_pages=[],
    )

    verdict = Director.synthesize_verdict(director, claim, research, hits=[])

    assert verdict.verdict == VerdictLabel.TRUE
    assert "Judge the claim exactly as written." in captured["prompt"]
    assert "Do NOT give a verdict solely for the embedded claim" in captured["prompt"]
