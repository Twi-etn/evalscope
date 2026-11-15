from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.io_utils import local_image_to_base64


@register_dataset('flickr8k')
class FlickrDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    Datasets: https://www.modelscope.cn/datasets/clip-benchmark/wds_flickr8k/files
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        dataset = self.load_local_dataset(split='test')

        for item in dataset:
            text = item['txt']
            base64_image = local_image_to_base64(item['jpg'])

            message = self.create_message(text=text, image_urls=base64_image)
            yield [message]
