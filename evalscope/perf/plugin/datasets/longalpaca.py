from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.dataset_args import TextDatasetArgs
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('longalpaca')
class LongAlpacaDatasetPlugin(DatasetPluginBase):
    """Read data from file which is list of requests.
    Sample: https://www.modelscope.cn/datasets/AI-ModelScope/LongAlpaca-12k/files
    """

    args_schema = TextDatasetArgs

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        path = self.local_dataset_path()
        if path.is_file() and path.suffix in {'.json', '.jsonl'}:
            ds = self.dataset_json_list(str(path))
        else:
            ds = self.load_local_dataset(split='train')
        for item in ds:
            prompt = item['instruction'].strip()
            result = self.prepare_messages(prompt)
            if result is None:
                continue
            yield result
