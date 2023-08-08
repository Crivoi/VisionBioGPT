"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/print.py"""
import datetime
import logging
import sys
import time

from settings.files import make_sure_parent_dir_exists

logger = logging.getLogger(__name__)

"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/print.py#L10"""


def print_seconds(seconds):
    msec = int(abs(seconds - int(seconds)) * 100)
    return f"{datetime.timedelta(seconds=int(seconds))}.{msec:02d}"


"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/print.py#L55"""


def log_metrics(split, metrics):
    logger.info(f"***** {split} metrics *****")
    metrics_formatted = metrics.copy()
    for k, v in metrics_formatted.items():
        if "_memory_" in k:
            metrics_formatted[k] = f"{v >> 20}MB"
        elif k.endswith("_runtime"):
            metrics_formatted[k] = print_seconds(v)
        elif k == "total_flos":
            metrics_formatted[k] = f"{int(v) >> 30}GF"
        elif isinstance(metrics_formatted[k], float):
            metrics_formatted[k] = round(v, 4)
    k_width = max(len(str(x)) for x in metrics_formatted.keys())
    v_width = max(len(str(x)) for x in metrics_formatted.values())
    for k in sorted(metrics_formatted.keys()):
        logger.info(f"  {k: <{k_width}} = {metrics_formatted[k]:>{v_width}}")


"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/print.py#L74"""


def print_large_integer(number, suffix=None):
    if suffix is None:
        if number < 1e3: return f"{number}"
        str_number = str(number)
        if number < 1e6: return f"{str_number[:-3]},{str_number[-3:]}"
        if number < 1e9: return f"{str_number[:-6]},{str_number[-6:-3]},{str_number[-3:]}"
        raise ValueError(f"Maybe not a good idea to display such a large number ({number}) in this way")
    else:
        if suffix == "B": return f"{float(number) / 1e9:.1f}B"
        if suffix == "M": return f"{float(number) / 1e6:.1f}M"
        if suffix == "K": return f"{float(number) / 1e3:.1f}K"

        if number < 1e3: return f"{number}"
        if number < 1e6: return f"{float(number) / 1e3:.1f}K"
        if number < 1e9: return f"{float(number) / 1e6:.1f}M"
        return f"{float(number) / 1e9:.1f}B"


"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/print.py#L103"""


def set_logging_format(log_filepath=None, debug=False):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_filepath is not None:
        make_sure_parent_dir_exists(log_filepath)
        handlers.append(logging.FileHandler(filename=log_filepath))

    if debug:
        logging.basicConfig(format="%(asctime)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                            level=logging.DEBUG, handlers=handlers)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=handlers)


"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/print.py#L117"""


def estimate_remaining_time(current_step, total_step, start_time):
    ratio = float(current_step) / total_step
    elapsed = time.time() - start_time
    if current_step == 0: return 0, elapsed, 0

    remaining = elapsed * (1 - ratio) / ratio
    return ratio, elapsed, remaining


"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/print.py#L127"""


def log_remaining_time(current_step, total_step, start_time, prefix="", suffix=""):
    ratio, elapsed, remaining = estimate_remaining_time(current_step, total_step, start_time)
    logger.info(f"{prefix}Progress: {current_step}/{total_step} ({ratio * 100:.1f}%); "
                f"Elapsed: {print_seconds(elapsed)}; Estimated remaining: {print_seconds(remaining)}{suffix}")


"""https://github.com/coastalcph/trldc/blob/main/dainlp/utils/print.py#L134"""


def speed_metrics(split, start_time, num_samples=None, num_steps=None):
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        result[f"{split}_samples_per_second"] = round(num_samples / runtime, 3)
    if num_steps is not None:
        result[f"{split}_steps_per_second"] = round(num_steps / runtime, 3)
    return result
