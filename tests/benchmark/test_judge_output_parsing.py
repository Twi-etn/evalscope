from typing import Optional

from evalscope.benchmarks.aime.aime_adapter import EQUIVALENCE_CONTRACT


class StubJudge:

    model_id = 'stub-judge'

    def __init__(self, response: str) -> None:
        self.response = response

    def judge(self, prompt: str) -> str:
        return self.response


def test_aime_llm_judge_accepts_only_a_json_verdict() -> None:
    """The judge now replies with JSON; a malformed verdict excludes the sample from the metric
    instead of being counted as an incorrect answer."""
    assert EQUIVALENCE_CONTRACT.parse('{"verdict": "Yes"}').value.verdict == 'Yes'

    assert not EQUIVALENCE_CONTRACT.parse('No, they are not equivalent. Yes would be wrong.').ok


def test_aime_verdict_cannot_be_read_out_of_prose() -> None:
    def parse(response: str) -> Optional[str]:
        result = EQUIVALENCE_CONTRACT.parse(response)
        return result.value.verdict if result.ok else None

    assert parse('```json\n{"verdict": "Yes"}\n```') is None
    assert parse('<think>Is 3/2 == 1.5? yes it is.</think>\n{"verdict": "Yes"}') is None

    # Every case below used to set the verdict by the word "Yes" appearing somewhere (#1578).
    assert parse('Yes') is None
    assert parse('Yes.') is None
    assert parse('**Yes**') is None
    assert parse('Yes, the answer is incorrect.') is None
    assert parse('Yes\n(give benefit of the doubt to units)') is None
    assert parse('[ERROR] judge request failed with status 500') is None
    # Two JSON objects that disagree are a parse failure rather than a silent pick.
    assert parse('```json\n{"verdict": "Yes"}\n```\n```json\n{"verdict": "No"}\n```') is None
