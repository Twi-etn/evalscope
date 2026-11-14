"""Export representative benchmark inputs as complete, gold-free decision perf requests."""

import argparse
import json
from pathlib import Path

from evalscope import TaskConfig
from evalscope.api.model.decision import DecisionInput
from evalscope.api.registry import get_benchmark
from evalscope.models.utils.decision import build_autotrust_request, build_systemone_request


def main() -> None:
    """Keep request sets and their lengths identical across concurrency levels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', choices=['systemone', 'autotrust'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument(
        '--choice-benchmark', choices=['contract_nli', 'hellaswag'], default='contract_nli',
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for kind, dataset in [('choice', args.choice_benchmark), ('noul', 'open_decisions'), ('score', 'stsb')]:
        config = TaskConfig(model=args.model, eval_type=args.api + '_api', datasets=[dataset],
                            limit=None if kind == 'noul' else args.samples)
        adapter = get_benchmark(dataset, config)
        bodies = []
        for samples in adapter.load_dataset().values():
            for sample in samples:
                user = next(message for message in sample.input if message.role == 'user')
                question = DecisionInput.model_validate(user.internal['systemone'])
                if question.kind != kind:
                    continue
                body = (
                    build_systemone_request(question, args.model)
                    if args.api == 'systemone'
                    else build_autotrust_request(question)
                )
                bodies.append(json.dumps(body, ensure_ascii=False))
        path = args.output / f'{kind}.jsonl'
        path.write_text('\n'.join(bodies[:args.samples]) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
