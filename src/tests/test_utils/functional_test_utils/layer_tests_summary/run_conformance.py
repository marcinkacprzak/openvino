# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from subprocess import Popen
from shutil import copytree, rmtree
from summarize import create_summary, create_api_summary
from merge_xmls import merge_xml
from run_parallel import TestParallelRunner
from pathlib import Path

import defusedxml.ElementTree as ET
from urllib.parse import urlparse

import os
import csv
import urllib.request as ur
from utils import constants
from utils.conformance_utils import get_logger
from utils import file_utils

logger = get_logger('conformance_runner')
has_python_api = True
try:
    from rename_conformance_ir import create_hash, save_rel_weights
except:
    logger.warning("Please set the above env variable to get the same conformance ir names run by run!")
    has_python_api = False

API_CONFORMANCE_BIN_NAME = "apiConformanceTests"
OP_CONFORMANCE_BIN_NAME = "conformanceTests"
SUBGRAPH_DUMPER_BIN_NAME = "subgraphsDumper"

SCRIPT_DIR_PATH, SCRIPT_NAME = os.path.split(os.path.abspath(__file__))
NO_MODEL_CONSTANT = os.path.join(SCRIPT_DIR_PATH, "data", "models.lst")

def get_default_working_dir():
    path = Path(__file__).parent.resolve()
    return os.path.join(path, "temp")

def parse_arguments():
    parser = ArgumentParser()

    models_path_help = "Path to the directory/ies containing models to dump subgraph (the default way is to download conformance IR). It may be directory, archieve file, .lst file with models to download by a link, model file paths. If --s=0, specify the Conformance IRs directory. NOTE: Applicable only for Opset Conformance."
    dump_graph_help = "Set '1' to create Conformance IRs from models using subgraphsDumper tool. The default value is '0'. NOTE: Applicable only for Opset Conformance."
    device_help = "Specify a target device. The default value is `CPU`"
    ov_help = "OV binary path. The default way is to find the absolute path of latest bin in the repo (by using script path)"
    working_dir_help = "Specify a working directory to save a run artifacts"
    type_help = "Specify conformance type: `OP` or `API`. The default value is `OP`"
    workers_help = "Specify number of workers to run in parallel. The default value is `CPU_count-1`"
    gtest_filter_helper = "Specify gtest filter to apply for a test run. E.g. *Add*:*BinaryConv*. The default value is None"
    ov_config_path_helper = "Specify path to a plugin config file as `.lst` file. Default value is ``"
    special_mode_help = "Specify shape mode (`static`, `dynamic` or ``) for Opset conformance or API scope type (`mandatory` or ``). Default value is ``"
    parallel_help = "Parallel over HW devices. For example run tests over GPU.0 and GPU.1 in case when device are the same"
    expected_failures_help = "Excepted failures list file path as csv"
    cache_path_help = "Path to the cache file with test_name list sorted by execution time as `.lst` file!"
    expected_failures_update_help = "Overwrite expected failures list in case same failures were fixed"

    parser.add_argument("-d", "--device", help= device_help, type=str, required=False, default="CPU")
    parser.add_argument("-t", "--type", help=type_help, type=str, required=False, default=constants.OP_CONFORMANCE)
    parser.add_argument("--gtest_filter", help=gtest_filter_helper, type=str, required=False, default="*")
    parser.add_argument("-w", "--working_dir", help=working_dir_help, type=str, required=False, default=get_default_working_dir())
    parser.add_argument("-m", "--models_path", help=models_path_help, type=str, required=False, default=NO_MODEL_CONSTANT)
    parser.add_argument("-ov", "--ov_path", help=ov_help, type=str, required=False, default="")
    parser.add_argument("-j", "--workers", help=workers_help, type=int, required=False, default=os.cpu_count()-1)
    parser.add_argument("-c", "--ov_config_path", help=ov_config_path_helper, type=str, required=False, default="")
    parser.add_argument("-s", "--dump_graph", help=dump_graph_help, type=int, required=False, default=0)
    parser.add_argument("-sm", "--special_mode", help=special_mode_help, type=str, required=False, default="")
    parser.add_argument("-p", "--parallel_devices", help=parallel_help, type=bool, required=False, default=False)
    parser.add_argument("-f", "--expected_failures", help=expected_failures_help, type=str, required=False, default="")
    parser.add_argument("-u", "--expected_failures_update", help=expected_failures_update_help, type=bool, required=False, default=False)
    parser.add_argument("--cache_path", help=cache_path_help, type=str, required=False, default="")

    return parser.parse_args()

class Conformance:
    def __init__(self, device:str, model_path:os.path, ov_path:os.path, type:str, workers:int,
                 gtest_filter:str, working_dir:os.path, ov_config_path:os.path, special_mode:str,
                 cache_path:str, parallel_devices:bool, expected_failures_file: str,
                 expected_failures_update: bool):
        self._device = device
        self._model_path = model_path
        if os.path.isdir(ov_path):
            self._ov_path = ov_path
        else:
            self._ov_path = file_utils.get_ov_path(SCRIPT_DIR_PATH, None, True)
        self._working_dir = working_dir
        if os.path.exists(self._working_dir):
            logger.info(f"Working dir {self._working_dir} is cleaned up")
            rmtree(self._working_dir)
        os.mkdir(self._working_dir)
        self._cache_path = cache_path if os.path.isfile(cache_path) else ""
        if type == constants.OP_CONFORMANCE:
            if special_mode == "static" or special_mode == "dynamic" or special_mode == "":
                self._special_mode = special_mode
            else:
                logger.error(f'Incorrect value to set shape mode: {special_mode}. Please check to get possible values')
                exit(-1)
            self._gtest_filter = gtest_filter
        elif type == constants.API_CONFORMANCE:
            self._special_mode = ""
            if special_mode == "mandatory":
                self._gtest_filter = f"*mandatory*{gtest_filter}*:*{gtest_filter}*mandatory*"
            elif special_mode == "":
                self._gtest_filter = gtest_filter
            else:
                logger.error(f'Incorrect value to set API scope: {special_mode}. Please check to get possible values')
                exit(-1)
        else:
            logger.error(f"Incorrect conformance type: {type}. Please use '{constants.OP_CONFORMANCE}' or '{constants.API_CONFORMANCE}'")
            exit(-1)
        self._type = type
        self._workers = workers
        if not os.path.exists(ov_config_path) and ov_config_path != "":
            logger.error(f"Specified config file does not exist: {ov_config_path}.")
            exit(-1)
        self._ov_config_path = ov_config_path
        self._is_parallel_over_devices = parallel_devices
        self._expected_failures = set()
        self._expected_failures_file = expected_failures_file
        if os.path.isfile(expected_failures_file):
            self._expected_failures = self.__get_failed_test_from_csv(expected_failures_file)
        else:
            logger.warning(f"Expected failures testlist `{self._expected_failures_file}` does not exist!")
        self._expected_failures_update = expected_failures_update

    def __download_models(self, url_to_download, path_to_save):
        _, file_name = os.path.split(urlparse(url_to_download).path)
        download_path = os.path.join(path_to_save, file_name)
        try:
            logger.info(f"Conformance IRs will be downloaded from {url_to_download} to {download_path}")
            ur.urlretrieve(url_to_download, filename=download_path)
        except:
            logger.error(f"Please verify URL: {url_to_download}. Looks like that is incorrect")
            exit(-1)
        logger.info(f"Conformance IRs were downloaded from {url_to_download} to {download_path}")
        if not os.path.isfile(download_path):
            logger.error(f"{download_path} is not a file. Exit!")
            exit(-1)
        if file_utils.is_archieve(download_path):
            logger.info(f"The file {download_path} is archieve. Should be unzip to {path_to_save}")
            return file_utils.unzip_archieve(download_path, path_to_save)
        return download_path
        

    def __dump_subgraph(self):
        subgraph_dumper_path = os.path.join(self._ov_path, f'{SUBGRAPH_DUMPER_BIN_NAME}{constants.OS_BIN_FILE_EXT}')
        if not os.path.isfile(subgraph_dumper_path):
            logger.error(f"{subgraph_dumper_path} is not exist!")
            exit(-1)
        conformance_ir_path = os.path.join(self._working_dir, "conformance_ir")
        if os.path.isdir(conformance_ir_path):
            logger.info(f"Remove directory {conformance_ir_path}")
            rmtree(conformance_ir_path)
        os.mkdir(conformance_ir_path)
        self._model_path = file_utils.prepare_filelist(self._model_path, ["*.onnx", "*.pdmodel", "*.__model__", "*.pb", "*.xml", "*.tflite"])
        logger.info(f"Stating model dumping from {self._model_path}")
        cmd = f'{subgraph_dumper_path} --input_folders="{self._model_path}" --output_folder="{conformance_ir_path}"'
        process = Popen(cmd, shell=True)
        out, err = process.communicate()
        if err is None:
            for line in str(out).split('\n'):
                logger.info(line)
            logger.info(f"Conformance IRs were saved to {conformance_ir_path}")
        else:
            logger.error(err)
            logger.error("Process failed on step: 'Subgraph dumping'")
            exit(-1)
        self._model_path = conformance_ir_path
        if has_python_api:
            op_rel_weight = create_hash(Path(self._model_path))
            save_rel_weights(Path(self._model_path), op_rel_weight)
            logger.info(f"All conformance IRs in {self._model_path} were renamed based on hash")
        else:
            logger.warning("The OV Python was not built or Environment was not updated to requirments. Skip the step to rename Conformance IR based on a hash")


    @staticmethod
    def __get_failed_test_from_csv(csv_file:str):
        failures = set()
        with open(csv_file, "r") as failures_file:
            for row in csv.reader(failures_file, delimiter=','):
                if row[0] == "Test Name":
                    continue
                failures.add(row[0])
            failures_file.close()
        return failures

    def __check_expected_failures(self):
        this_failures_file = os.path.join(self._working_dir, f"{self._device}_logs", "logs", "fix_priority.csv")
        if not os.path.isfile(this_failures_file):
            return
        this_run_failures = self.__get_failed_test_from_csv(this_failures_file)

        diff = this_run_failures.difference(self._expected_failures)
        if len(diff) > 0:
            logger.error(f"Unexpected failures: {diff}")
            exit(-1)
        
        intersection = self._expected_failures.intersection(this_run_failures)
        if this_run_failures != self._expected_failures and self._expected_failures_update:
            logger.info(f"Expected failures file {self._expected_failures} will be updated!!!")
            os.remove(self._expected_failures_file)
            this_failures_file = Path(this_failures_file)
            this_failures_file.rename(self._expected_failures_file)

    def __run_conformance(self):
        conformance_path = None
        if self._type == constants.OP_CONFORMANCE:
            conformance_path = os.path.join(self._ov_path, f'{OP_CONFORMANCE_BIN_NAME}{constants.OS_BIN_FILE_EXT}')
        else:
            conformance_path = os.path.join(self._ov_path, f'{API_CONFORMANCE_BIN_NAME}{constants.OS_BIN_FILE_EXT}')

        if not os.path.isfile(conformance_path):
            logger.error(f"{conformance_path} is not exist!")
            exit(-1)

        logs_dir = os.path.join(self._working_dir, f'{self._device}_logs')
        report_dir = os.path.join(self._working_dir, 'report')
        if os.path.isdir(report_dir):
            logger.info(f"Report dir {report_dir} is cleaned up")
            rmtree(report_dir)
        parallel_report_dir = os.path.join(report_dir, 'parallel')
        if not os.path.isdir(report_dir):
            os.mkdir(report_dir)
        if not os.path.isdir(logs_dir):
            os.mkdir(logs_dir)
        
        command_line_args = [f"--device={self._device}", 
                             f'--input_folders="{self._model_path}"' if self._type == constants.OP_CONFORMANCE else '',
                             f"--report_unique_name", f'--output_folder="{parallel_report_dir}"',
                             f'--gtest_filter=\"{self._gtest_filter}\"', f'--config_path="{self._ov_config_path}"',
                             f'--shape_mode={self._special_mode}']
        conformance = TestParallelRunner(f"{conformance_path}",
                                         command_line_args,
                                         self._workers,
                                         logs_dir,
                                         self._cache_path,
                                         self._is_parallel_over_devices,
                                         self._expected_failures if not self._expected_failures_update else set())
        conformance.run()
        conformance.postprocess_logs()

        if os.path.isfile(self._expected_failures_file):
            self.__check_expected_failures()

        final_report_name = f'report_{self._type.lower()}'
        merge_xml([parallel_report_dir], report_dir, final_report_name, self._type, True)

        logger.info(f"Conformance is successful. XML reportwas saved to {report_dir}")
        return (os.path.join(report_dir, final_report_name + ".xml"), report_dir)

    def __summarize(self, xml_report_path:os.path, report_dir: os.path):
        if self._type == constants.OP_CONFORMANCE:
            summary_root = ET.parse(xml_report_path).getroot()
            rel_weights_path = os.path.join(self._model_path, constants.REL_WEIGHTS_FILENAME.replace(constants.REL_WEIGHTS_REPLACE_STR, self._special_mode))
            create_summary(summary_root, report_dir, [], "", "", True, True, rel_weights_path)
        else:
            create_api_summary([xml_report_path], report_dir, [], "", "")
        copytree(os.path.join(SCRIPT_DIR_PATH, "template"), os.path.join(report_dir, "template"))
        logger.info(f"Report was saved to {os.path.join(report_dir, 'report.html')}")

    def run(self, dump_models: bool):
        command = f'{constants.PIP_NAME} install -r {os.path.join(SCRIPT_DIR_PATH, "requirements.txt")}'
        process = Popen(command, shell=True)
        out, err = process.communicate()
        if err is None:
            if not out is None:
                for line in str(out).split('\n'):
                    logger.info(line)
        else:
            logger.error(err)
            logger.error("Impossible to install requirements!")
            exit(-1)
        logger.info(f"[ARGUMENTS] --device = {self._device}")
        logger.info(f"[ARGUMENTS] --type = {self._type}")
        logger.info(f"[ARGUMENTS] --working_dir = {self._working_dir}")
        logger.info(f"[ARGUMENTS] --ov_path = {self._ov_path}")
        logger.info(f"[ARGUMENTS] --ov_config_path = {self._ov_config_path}")
        logger.info(f"[ARGUMENTS] --workers = {self._workers}")
        logger.info(f"[ARGUMENTS] --gtest_filter = {self._gtest_filter}")
        logger.info(f"[ARGUMENTS] --models_path = {self._model_path}")
        logger.info(f"[ARGUMENTS] --dump_graph = {dump_models}")
        logger.info(f"[ARGUMENTS] --shape_mode = {self._special_mode}")
        logger.info(f"[ARGUMENTS] --parallel_devices = {self._is_parallel_over_devices}")
        logger.info(f"[ARGUMENTS] --cache_path = {self._cache_path}")
        logger.info(f"[ARGUMENTS] --expected_failures = {self._expected_failures_file}")
        logger.info(f"[ARGUMENTS] --expected_failures_update = {self._expected_failures_update}")

        if self._type == constants.OP_CONFORMANCE:
            if file_utils.is_url(self._model_path):
                self._model_path = self.__download_models(self._model_path, self._working_dir)
            if self._model_path == NO_MODEL_CONSTANT or os.path.splitext(self._model_path)[1] == ".lst":
                with open(self._model_path, "r") as model_list_file:
                    model_dir = os.path.join(self._working_dir, "models")
                    if not os.path.isdir(model_dir):
                        os.mkdir(model_dir)
                    for model in model_list_file.readlines():
                        self.__download_models(model, model_dir)
                    self._model_path = model_dir
            if dump_models:
                self.__dump_subgraph()
            if not os.path.exists(self._model_path):
                logger.error(f"The model direstory {self._model_path} does not exist!")
                exit(-1)
            if not os.path.exists(self._model_path):
                logger.error(f"Directory {self._model_path} does not exist")
                exit(-1)
        xml_report, report_dir = self.__run_conformance()
        self.__summarize(xml_report, report_dir)

if __name__ == "__main__":
    args = parse_arguments()
    conformance = Conformance(args.device, args.models_path,
                              args.ov_path, args.type,
                              args.workers, args.gtest_filter,
                              args.working_dir, args.ov_config_path,
                              args.special_mode, args.cache_path,
                              args.parallel_devices, args.expected_failures,
                              args.expected_failures_update)
    conformance.run(args.dump_graph)
    
