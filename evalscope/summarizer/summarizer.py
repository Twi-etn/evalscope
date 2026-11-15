# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import json
import os
from typing import List, Union

from evalscope.config import TaskConfig, parse_task_config
from evalscope.constants import EvalBackend
from evalscope.report import gen_table
from evalscope.utils.io_utils import OutputsStructure, csv_to_list, get_latest_folder_path, json_to_dict, yaml_to_dict
from evalscope.utils.logger import get_logger

logger = get_logger()


class Summarizer:
    @staticmethod
    def get_report(outputs_dir: str) -> List[dict]:
        res_list: list = []

        outputs_structure = OutputsStructure(outputs_dir, is_make=False)
        reports_dir: str = outputs_structure.reports_dir
        if reports_dir is None:
            raise ValueError(f'No reports directory in {outputs_dir}')

        report_files: list = glob.glob(os.path.join(reports_dir, '**/*.json'))
        for report_file in report_files:
            with open(report_file, 'r', encoding='utf-8') as f:
                res_list.append(json.load(f))

        report_table: str = gen_table(reports_path_list=[reports_dir])
        logger.info(f'*** Report table ***\n{report_table}')

        return res_list

    @staticmethod
    def get_report_from_cfg(task_cfg: Union[str, List[str], TaskConfig, List[TaskConfig], dict]) -> List[dict]:
        """
        Get report from cfg file.

        Args:
            task_cfg: task cfg file path. refer to evalscope/tasks/eval_qwen-7b-chat_v100.yaml

        Returns:
            list: list of report dict.
            A report dict is overall report on a benchmark for specific model.
        """
        final_res_list: List[dict] = []
        candidate_task_cfgs: List[TaskConfig] = []

        if isinstance(task_cfg, list):
            for task_cfg_item in task_cfg:
                candidate_task_cfgs.append(parse_task_config(task_cfg_item))
        else:
            candidate_task_cfgs.append(parse_task_config(task_cfg))

        for candidate_task in candidate_task_cfgs:
            logger.info(f'**Loading task cfg for summarizer: {candidate_task}')
            eval_backend = candidate_task.eval_backend

            if eval_backend == EvalBackend.NATIVE:
                outputs_dir: str = os.path.expanduser(candidate_task.work_dir)
                if outputs_dir is None:
                    raise ValueError(f'No outputs_dir in {task_cfg}')
                res_list: list = Summarizer.get_report(outputs_dir=outputs_dir)
                final_res_list.extend(res_list)

            elif eval_backend == EvalBackend.THIRD_PARTY:
                raise ValueError('*** The summarizer for Third party evaluation backend is not supported yet ***')
            else:
                raise ValueError(f'Invalid eval_backend: {eval_backend}')

        return final_res_list


if __name__ == '__main__':
    cfg_file = 'registry/tasks/eval_qwen-7b-chat_v100.yaml'
    report_list = Summarizer.get_report_from_cfg(cfg_file)

    print(report_list)
