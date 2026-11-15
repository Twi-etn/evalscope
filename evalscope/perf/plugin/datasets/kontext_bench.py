from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.io_utils import local_image_to_base64


@register_dataset('kontext_bench')
class KontextDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    Datasets: https://modelscope.cn/datasets/black-forest-labs/kontext-bench/dataPeview
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        dataset = self.load_local_dataset(split='test')

        for item in dataset:
            text = item.get('instruction') or 'Describe the image'
            base64_image = local_image_to_base64(item['image'])

            message = self.create_message(text=text, image_urls=base64_image)
            yield [message]
