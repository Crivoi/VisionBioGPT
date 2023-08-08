"""
https://github.com/coastalcph/trldc/blob/main/dainlp/utils/files.py
"""
import json
import logging
import os
import re
import shutil
from pathlib import Path

import numpy

logger = logging.getLogger(__name__)

"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/files.py#L9"""


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/files.py#L22"""


def make_sure_parent_dir_exists(filepath):
    parent_dir = os.path.dirname(filepath)
    os.makedirs(parent_dir, exist_ok=True)


"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/files.py#L37"""


def write_object_to_json_file(data, filepath, sort_keys=False):
    make_sure_parent_dir_exists(filepath)
    json.dump(data, open(filepath, "w"), indent=2, sort_keys=sort_keys, default=lambda o: "Unknown")


"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/files.py#L53"""


def remove_checkpoints(output_dir, prefix, save_total_limit, best_model_checkpoint=None):
    if save_total_limit is None or save_total_limit <= 0: return
    checkpoints_sorted = sorted_checkpoints(output_dir, prefix, best_model_checkpoint)
    if len(checkpoints_sorted) <= save_total_limit: return
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint)


"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/files.py#L65"""


def sorted_checkpoints(output_dir, prefix, best_model_checkpoint):
    ignored = [] if best_model_checkpoint is None else [str(Path(best_model_checkpoint))]

    checkpoints = []
    for path in Path(output_dir).glob(f"{prefix}-*"):  # the folders have name like 'checkpoint-15'
        if str(path) in ignored: continue
        regex_match = re.match(f".*{prefix}-([0-9]+)", str(path))
        assert regex_match is not None and regex_match.groups() is not None
        checkpoints.append((int(regex_match.groups()[0]), str(path)))

    return [c[1] for c in sorted(checkpoints)]
