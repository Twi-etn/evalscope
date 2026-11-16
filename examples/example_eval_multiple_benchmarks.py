"""Evaluate one model on three local benchmarks in the same run.

Place math_500.lance, gpqa_diamond.lance, and aime24.lance in ./datasets
or point EVALSCOPE_DATA_DIR to the directory holding them.
"""

import os
from pathlib import Path

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType


def main() -> None:
    """Run each benchmark with its own dataset configuration."""
    data_dir = Path(os.getenv('EVALSCOPE_DATA_DIR', 'datasets')).expanduser()
    task_cfg = TaskConfig(
        model=os.getenv('EVALSCOPE_MODEL', 'DeepSeek-R1-Distill-Qwen-1.5B'),
        api_url=os.getenv('EVALSCOPE_API_URL', 'http://127.0.0.1:8000/v1/chat/completions'),
        api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
        eval_type=EvalType.OPENAI_API,
        datasets=['math_500', 'gpqa_diamond', 'aime24'],
        dataset_args={
            'math_500': {
                'local_path': str(data_dir / 'math_500.lance'),
                'few_shot_num': 0,
                # 'subset_list': ['Level 1', 'Level 2'],  # Evaluate only these difficulty levels.
            },
            'gpqa_diamond': {
                'local_path': str(data_dir / 'gpqa_diamond.lance'),
                'few_shot_num': 0,
            },
            'aime24': {
                'local_path': str(data_dir / 'aime24.lance'),
                'dataset_revision': 1,  # Use a version ID from your dataset, or omit it for the latest version.
                'few_shot_num': 0,
            },
        },
        judge={'strategy': 'rule'},
        limit=5,  # Remove this line to evaluate every sample in each subset.
        eval_batch_size=4,
        generation_config={'max_tokens': 4096, 'temperature': 0.0},
        work_dir='outputs/multiple_benchmarks',
    )
    run_task(task_cfg)


if __name__ == '__main__':
    main()
