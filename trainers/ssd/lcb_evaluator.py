"""LiveCodeBench v6 evaluator ported from Apple ml-ssd.

This module ports the LiveCodeBench evaluation pipeline from Apple's ml-ssd project
(https://github.com/apple/ml-ssd) and adapts it to the auto-coder-trainer
BaseEvaluator interface.

Reference paper:
    Agarwal, R., et al. "Revisiting Rejection Sampling for Reinforcement Learning
    from Human Feedback." arXiv preprint arXiv:2409.09439 (2024).
    https://arxiv.org/abs/2409.09439

Utility functions and test execution logic are preserved from the original ml-ssd
implementation, which itself incorporates code from the official LiveCodeBench repo:
    https://github.com/LiveCodeBench/LiveCodeBench
"""
from __future__ import annotations

import ast
import base64
import copy
import faulthandler
import io
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import re
import sys
import textwrap
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from types import ModuleType
from typing import Any, Dict, List, Optional
from unittest.mock import mock_open, patch

import numpy as np

from evaluators.base import BaseEvaluator, BenchmarkResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_IMPORTS = """from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge
from functools import reduce, cache, lru_cache
from random import randrange, shuffle
from operator import itemgetter, sub
from re import search as re_search  # Assuming 're' refers to a regex search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import numpy as np
import pandas as pd
from math import log, prod  # 'log' and 'prod' are functions in the math module
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
from operator import iand
import sys
"""

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

LCB_PROMPT_WITHOUT_STARTER_CODE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
  # YOUR CODE HERE
```"""

LCB_PROMPT_WITH_STARTER_CODE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
```python
{entry_point}
"""

# ---------------------------------------------------------------------------
# Helper classes for stdin/stdout capture
# (from official LiveCodeBench testing_util.py)
# ---------------------------------------------------------------------------


class Capturing(list):
    """
    Context manager to capture stdout as a list.
    From official LiveCodeBench testing_util.py:59-70
    From https://stackoverflow.com/a/16571630/6416660
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class MockBuffer:
    """
    Mock for sys.stdin.buffer with byte string support.
    From official LiveCodeBench testing_util.py:94-103
    """
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")  # Convert to bytes

    def read(self, *args):
        # Return as byte strings that can be split
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    """
    Custom mock for sys.stdin that supports buffer attribute.
    From official LiveCodeBench testing_util.py:74-91
    """
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = io.StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __iter__(self):
        # Support `for line in sys.stdin` pattern
        return iter(self._stringio)

    def __next__(self):
        # Support `next(sys.stdin)` pattern
        return next(self._stringio)

    def __getattr__(self, name):
        # Delegate other attributes to StringIO
        return getattr(self._stringio, name)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def get_stripped_lines(val: str):
    """
    Strip the entire value and then strip each line individually.
    From official LiveCodeBench repo - ensures proper whitespace/newline handling.
    """
    ## you don't want empty lines to add empty list after splitlines!
    val = val.strip()
    return [val_line.strip() for val_line in val.split("\n")]


def convert_line_to_decimals(line: str) -> tuple:
    """
    Convert a line of space-separated values to Decimal objects for precise numeric comparison.
    From official LiveCodeBench repo testing_util.py:214-220
    Used for stdio tests where outputs are strings.

    Returns:
        (success: bool, decimals: list[Decimal])
    """
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def compare_strings_with_decimal_fallback(prediction_str: str, expected_str: str) -> bool:
    """
    Compare string outputs with Decimal fallback for numeric values.
    From official LiveCodeBench repo testing_util.py:372-423 (grade_stdio comparison logic)

    Returns True if outputs match (exact or via Decimal comparison).
    """
    # Use official's multi-stage stripping strategy
    stripped_prediction_lines = get_stripped_lines(prediction_str)
    stripped_expected_lines = get_stripped_lines(expected_str)

    # Check if line counts match
    if len(stripped_prediction_lines) != len(stripped_expected_lines):
        return False

    # Line-by-line comparison with exact match first, then decimal fallback
    for stripped_pred_line, stripped_exp_line in zip(stripped_prediction_lines, stripped_expected_lines):
        ## CASE 1: exact match
        if stripped_pred_line == stripped_exp_line:
            continue

        ## CASE 2: element-wise comparison using Decimal for precise float comparison
        success, decimal_pred_line = convert_line_to_decimals(stripped_pred_line)
        if not success:
            return False

        success, decimal_exp_line = convert_line_to_decimals(stripped_exp_line)
        if not success:
            return False

        if decimal_pred_line == decimal_exp_line:
            continue

        # If neither exact match nor decimal match worked, fail
        return False

    # All lines matched
    return True


def reliability_guard():
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def has_test_type(tests, type):  ## helper to select specific type of problems
    """
    Check if any test in the test list has 'testtype' set to 'type'.
    """
    test_list = json.loads(tests)
    for test in test_list:
        if test.get("testtype") == type:
            return True
    return False


def translate_private_test_cases(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    test_cases = json.loads(original_data)
    return test_cases


def map_to_example(row):
    # Parse metadata JSON string to dict (matches official LiveCodeBench code_generation.py:76)
    metadata_raw = row.get("metadata", "{}")
    try:
        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
    except json.JSONDecodeError:
        metadata = {}

    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "task_id": row["question_id"],
        "is_stdin": has_test_type(row["public_test_cases"], "stdin"),
        "public_test_cases": row["public_test_cases"],
        "difficulty": row["difficulty"],
        "metadata": metadata,
    }


def post_process_code(code):
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.replace("<code>", "")
    return code


def parse_function_name_from_starter_code(starter_code):
    """
    Extract function name from starter code using AST parsing.
    Based on official LiveCodeBench implementation.

    Args:
        starter_code: Python code string containing function definition

    Returns:
        Function name string, or None if not found
    """
    try:
        # Handle incomplete starter code by adding a pass statement if needed
        # LeetCode-style starter code often has incomplete function definitions
        code_to_parse = starter_code
        if not code_to_parse.strip().endswith(("pass", "...", "return")):
            # Count indentation of last line to add proper pass statement
            lines = code_to_parse.rstrip().split('\n')
            if lines:
                last_line = lines[-1]
                # If last line ends with ':', add indented pass
                if last_line.rstrip().endswith(':'):
                    indent = len(last_line) - len(last_line.lstrip()) + 4
                    code_to_parse = code_to_parse + '\n' + ' ' * indent + 'pass'

        tree = ast.parse(code_to_parse)
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # For LeetCode-style problems, there should be exactly one function
                # If there are multiple, we take the last one found (which is typically the target)
                fn = node.name
        return fn
    except Exception:
        return None


def clean_if_name(code: str) -> str:
    """
    Remove 'if __name__ == "__main__":' wrapper from code using AST parsing.
    From official LiveCodeBench testing_util.py:106-119

    The runtime doesn't interact well with __name__ == '__main__', so we unwrap it.
    Extracts the code inside the if block and returns it without the if wrapper.
    """
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except:
        pass

    return code


def make_function(code: str) -> str:
    """
    Wrap code inside a function for controlled execution.
    From official LiveCodeBench testing_util.py:122-151

    Separates imports from other statements and wraps non-import code in wrapped_function().
    This allows us to call the code as a function with proper stdin mocking.
    """
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            BASE_IMPORTS
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception as e:
        return code


def compile_code(code: str) -> ModuleType:
    """
    Compile code into a module.
    From official LiveCodeBench testing_util.py:192-211

    Note: Removed signal.alarm() calls since timeout is handled by parent process in lcb_run().
    """
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            # leetcode wraps solutions in `Solution`
            # this is a hack to check if it is leetcode solution or not
            # currently livecodebench only supports LeetCode but
            # else condition allows future extensibility to other platforms
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accesible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
        return compiled_sol
    except Exception:
        return None


def get_function(compiled_sol, fn_name: str):
    """
    Safely extract function from compiled module.
    From official LiveCodeBench testing_util.py:184-189
    """
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception as e:
        return None


def call_method(method, inputs):
    """
    Call method with comprehensive stdin mocking.
    From official LiveCodeBench testing_util.py:154-181

    Provides full stdin support including read(), readline(), readlines(), and buffer.
    """
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # Create custom stdin mock with buffer support
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)  # Use our custom mock instead of StringIO
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def prepare_test_input_output_std(test_case):
    """
    Prepare test input/output for stdin-based tests.

    From official LiveCodeBench testing_util.py:310-425 (grade_stdio flow).
    The strip() on output is critical for proper comparison - see compare_strings_with_decimal_fallback().

    Args:
        test_case: Dict with "input" (str) and "output" (str) keys

    Returns:
        (test_input: str, test_output: str) tuple
    """
    test_input = test_case["input"]
    test_output = test_case["output"].strip()
    return test_input, test_output


def run_test_func(completion, is_extracted, test_input, test_output, func_name):
    """
    Run function-based test with unified comparison logic.

    Difference from run_test_std: Only HOW prediction is obtained (function call vs stdio)
    Comparison logic: Same as run_test_std (string comparison with Decimal fallback)

    Args:
        completion: The code to execute
        is_extracted: Whether inputs are already extracted/parsed
        test_input: Test input data
        test_output: Expected output
        func_name: Name of the function to call (required)
    """
    assert func_name is not None, "func_name must be provided"

    namespace = {}
    exec(completion, namespace)

    # Detect if completion is class-based (e.g., class Solution:)
    is_class_based = "class Solution:" in completion or "class Solution(" in completion

    output = io.StringIO()
    sys.stdout = output

    try:
        # Get the callable (either function or method)
        if is_class_based:
            solution_instance = namespace["Solution"]()
            callable_func = getattr(solution_instance, func_name)
        else:
            callable_func = namespace[func_name]

        # Call the function/method with appropriate arguments
        if not is_extracted:
            if isinstance(test_input, dict):
                prediction = callable_func(**test_input)
            else:
                prediction = callable_func(test_input)
        else:
            prediction = callable_func(*test_input)

        # Don't penalize model if it produces tuples instead of lists
        if isinstance(prediction, tuple):
            prediction = list(prediction)

        # Convert both prediction and expected to strings for unified comparison
        prediction_str = str(prediction) if not isinstance(prediction, str) else prediction
        expected_str = str(test_output) if not isinstance(test_output, str) else test_output

        # Use unified comparison logic (same as run_test_std)
        if compare_strings_with_decimal_fallback(prediction_str, expected_str):
            return True, prediction
        else:
            return False, prediction

    except Exception as e:
        error_msg = f"Error: {str(e)}" if not is_extracted else str(e)
        return False, error_msg

    finally:
        sys.stdout = sys.__stdout__


def run_test_std(completion, test_input, test_output):
    """
    Run stdin-based test using official LiveCodeBench approach.
    Based on testing_util.py:310-425 (grade_stdio)

    Uses AST-based code transformation and comprehensive stdin mocking.
    """
    # Clean if __name__ == "__main__" wrapper
    completion = clean_if_name(completion)

    # Wrap code in function for controlled execution
    completion = make_function(completion)

    # Compile code into module
    compiled_sol = compile_code(completion)
    if compiled_sol is None:
        return False, "Compilation failed"

    # Get wrapped function
    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return False, "Could not find wrapped_function"

    # Execute with captured stdout
    with Capturing() as captured_output:
        try:
            call_method(method, test_input)
        except Exception as e:
            return False, f"Runtime error: {e}"

    prediction = captured_output[0] if captured_output else ""

    # Use unified comparison logic (same as official testing_util.py:372-423)
    if compare_strings_with_decimal_fallback(prediction, test_output):
        return True, prediction.strip()
    else:
        return False, prediction.strip()


def prepare_test_input_output_functional(test_case, is_extracted):
    if not is_extracted:
        # Extract input and expected output from JSON directly
        test_input = test_case["input"]
        test_output = test_case["output"]
        return test_input, test_output
    else:
        # Robustly process complex inputs
        input_str = test_case["input"]
        expected_output = test_case["output"].strip()
        inputs = []

        if "=" in input_str:
            parts = input_str.split(",") if "," in input_str else [input_str]
            for part in parts:
                key, value = map(str.strip, part.split("="))
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        value = value.strip('"')
                inputs.append(value)
        else:
            for line in input_str.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('"') and line.endswith('"'):
                    inputs.append(line.strip('"'))
                    continue
                if line.startswith("[") and line.endswith("]"):
                    inputs.append(json.loads(line))
                    continue
                try:
                    inputs.append(int(line))
                except ValueError:
                    try:
                        inputs.append(float(line))
                    except ValueError:
                        inputs.append(line)

        try:
            expected_output = json.loads(expected_output)
        except json.JSONDecodeError:
            expected_output = expected_output.strip()
        return inputs, expected_output


def run_tests_for_one_example(problem, test_cases, completion, result_list, is_extracted):
    time_elapsed = float("inf")
    test_type = test_cases[0]["testtype"]
    reliability_guard()
    completion = BASE_IMPORTS + "\n" + completion

    # Extract function name from metadata or parse from starter code
    func_name = None
    if test_type == "functional":
        # Try to get func_name from metadata first
        metadata = problem.get("metadata", {})
        func_name = metadata.get("func_name")

        # If not in metadata, parse from starter code
        if not func_name and "entry_point" in problem:
            func_name = parse_function_name_from_starter_code(problem["entry_point"])

    for i, test_case in enumerate(test_cases):
        output_error = ""
        output_value = ""
        try:
            time_start = time.time()
            if test_type == "functional":
                test_input, test_output = prepare_test_input_output_functional(test_case, is_extracted)
                passed, output_value = run_test_func(
                    completion, is_extracted, copy.deepcopy(test_input), copy.deepcopy(test_output), func_name
                )
            else:
                test_input, test_output = prepare_test_input_output_std(test_case)
                passed, output_value = run_test_std(completion, copy.deepcopy(test_input), copy.deepcopy(test_output))
            time_elapsed = time.time() - time_start
            if not passed:
                output_error = (
                    f"For test input: {test_input}. Expected output is: {test_output}, but got: {output_value}."
                )

        except Exception as e:
            passed = False
            output_error = f"For test input: {test_input}. Expected output is: {test_output}, but got error: {e}."
            output_value = f"Error: {e}."
        if output_error == "":
            output_error = f"For test input: {test_input}. Expected output is: {test_output}, your solution correctly passes this test with output {output_value}."
        result_list.append((passed, output_error, output_value, time_elapsed))
        if not passed:
            return


def lcb_run(problem, completion, timeout, is_extracted):
    test_cases = problem["test"]
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=run_tests_for_one_example, args=(problem, test_cases, completion, result, is_extracted))
    p.start()
    p.join(timeout=(timeout + 1) * len(test_cases) + 5)
    if p.is_alive():
        p.kill()

    # if len(result) < len(test_cases): failed due to timeout
    for i in range(len(test_cases) - len(result)):
        result.append((False, f"Time out!.", "Error: Time out!", float("inf")))
    return result


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def compute_metrics_from_results(results, k_list=[1, 5]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist()
        for k in ks
        if (total >= k).all()
    }
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k


# ---------------------------------------------------------------------------
# Functions from benchmark.py
# ---------------------------------------------------------------------------


def has_code(response):
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    return matches


def filter_by_contest_date(example):
    target_months = ["2025-02", "2025-03", "2025-04", "2025-05"]
    return example["contest_date"][:7] in target_months


# ---------------------------------------------------------------------------
# Internal LiveCodeBenchV6 benchmark runner
# ---------------------------------------------------------------------------


class _LiveCodeBenchV6:
    """
    LiveCodeBench V6 - Benchmark for evaluating code generation capabilities of LLMs
    on competitive programming problems from recent contests (Feb-May 2025).
    """

    def __init__(
        self,
        llm,
        tokenizer,
        max_tokens: int = 32768,
        n_repeat: int = 20,
        sampling_params: Optional[Dict[str, Any]] = None,
        seed: Optional[List[int]] = None,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.n_repeat = n_repeat
        self.sampling_params = sampling_params if sampling_params is not None else {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        }
        self.seed = seed if seed is not None else [0, 1234, 1234, 1234]
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        """Run the full benchmark: load data, generate solutions, evaluate."""
        ds = self.load_questions()
        examples = list(ds)
        self.generate(examples)
        results = self.evaluate(examples)
        return results

    def generate(self, examples):
        """Generate solution completions using vLLM."""
        from vllm import SamplingParams

        all_outputs = []
        stop_token_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        for i in range(self.n_repeat):
            seed = self.seed[0] + i

            prompts = []
            for example in examples:
                if example["is_stdin"]:
                    prompt_text = LCB_PROMPT_WITHOUT_STARTER_CODE.format(
                        problem_description=example["prompt"]
                    )
                else:
                    prompt_text = LCB_PROMPT_WITH_STARTER_CODE.format(
                        problem_description=example["prompt"],
                        entry_point=example["entry_point"],
                    )

                messages = [{"role": "user", "content": prompt_text}]
                templated = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(templated)

            sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                seed=seed,
                stop_token_ids=stop_token_ids,
                **self.sampling_params,
            )

            self.logger.info(f"Generating responses (repeat {i + 1}/{self.n_repeat})...")
            outputs = self.llm.generate(prompts, sampling_params)
            texts = [o.outputs[0].text for o in outputs]
            all_outputs.append(texts)

        for example, per_example_outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(per_example_outputs)
            example["model_answers"] = [has_code(o) for o in per_example_outputs]

    @staticmethod
    def check_correctness(problem: Dict, completion: str, timeout: float, is_extracted: bool = False) -> Dict:
        """Evaluate functional correctness by running the test suite."""
        result_list = lcb_run(problem, completion, timeout, is_extracted)
        details = [r[0] for r in result_list]
        all_passed = all(details)
        return {
            "all_passed": all_passed,
            "result_list": result_list,
            "test_cases": problem["test"],
        }

    def evaluate_single_example(self, example):
        """Evaluate a single example by running its code against test cases."""
        try:
            response_entry = {
                "task_id": example.get("task_id"),
                "prompt": example.get("prompt", ""),
                "entry_point": example.get("entry_point", ""),
                "is_stdin": example.get("is_stdin", False),
                "content": example["model_answer"],
                "difficulty": example["difficulty"],
                "correctness": None,
                "reason": None,
                "test_input": None,
                "test_output": None,
                "test_expected": None,
                "num_tests_passed": 0,
                "num_tests_failed": 0,
                "test_results": [],
            }

            code_filter_result = example["model_answer"]

            if not code_filter_result or len(code_filter_result) == 0:
                response_entry["correctness"] = False
                response_entry["reason"] = "Does not contain code component."
                return response_entry

            try:
                last_code = code_filter_result[-1]
                problem_to_check = copy.deepcopy(example)

                self.logger.debug(f"Evaluating {example['difficulty']} problem...")

                curr_res = self.check_correctness(
                    problem=problem_to_check,
                    completion=post_process_code(last_code),
                    timeout=6,
                    is_extracted=not problem_to_check["is_stdin"],
                )

                self.logger.debug(f"Result for {example['difficulty']}: {curr_res['all_passed']}")

                result_list = curr_res["result_list"]
                test_cases = curr_res["test_cases"]

                num_passed = sum(1 for r in result_list if r[0])
                num_failed = len(result_list) - num_passed

                response_entry["test_results"] = [1 if r[0] else 0 for r in result_list]
                response_entry["num_tests_passed"] = num_passed
                response_entry["num_tests_failed"] = num_failed
                response_entry["correctness"] = curr_res["all_passed"]

                if not curr_res["all_passed"]:
                    response_entry["reason"] = "Code is incorrect."

                    for idx, (passed, output_error, output_value, time_elapsed) in enumerate(result_list):
                        if not passed and idx < len(test_cases):
                            test_case = test_cases[idx]
                            response_entry["test_input"] = str(test_case.get("input", ""))
                            response_entry["test_expected"] = str(test_case.get("output", ""))
                            response_entry["test_output"] = str(output_value)
                            break
                else:
                    response_entry["reason"] = ""

            except Exception as e:
                self.logger.error(f"Error evaluating {example['difficulty']} example: {str(e)}")
                response_entry["correctness"] = False
                response_entry["reason"] = f"Evaluation error: {str(e)}"

            return response_entry

        except Exception as outer_e:
            self.logger.error(f"Outer error in evaluate_single_example: {str(outer_e)}")
            return {
                "task_id": example.get("task_id"),
                "prompt": example.get("prompt", ""),
                "entry_point": example.get("entry_point", ""),
                "is_stdin": example.get("is_stdin", False),
                "content": example.get("model_answer"),
                "difficulty": example.get("difficulty"),
                "correctness": False,
                "reason": f"Critical error: {str(outer_e)}",
                "test_input": None,
                "test_output": None,
                "test_expected": None,
                "num_tests_passed": 0,
                "num_tests_failed": 0,
                "test_results": [],
            }

    def evaluate(self, examples):
        """Evaluate generated solutions using parallel thread execution."""
        self.logger.info(f"Evaluating {len(examples)} examples...")
        self.logger.warning("Expect some output leaks from code/test execution into stdout")

        # Organize completions by repeat index
        examples_by_repeat = defaultdict(list)
        for example in examples:
            for i, (output, answers) in enumerate(zip(example["model_outputs"], example["model_answers"])):
                example_copy = example.copy()
                example_copy["model_answer"] = answers
                example_copy["model_output"] = output
                example_copy.pop("model_outputs", None)
                example_copy.pop("model_answers", None)
                examples_by_repeat[i].append(example_copy)

        all_repeat_results = []
        num_questions = len(examples)

        for repeat_idx, repeat_examples in examples_by_repeat.items():
            results = []
            with ThreadPoolExecutor(max_workers=32) as executor:
                future_to_example = {}
                for i, example in enumerate(repeat_examples):
                    future = executor.submit(self.evaluate_single_example, example)
                    future_to_example[future] = (i, example)

                results = [None] * len(repeat_examples)
                for future in as_completed(future_to_example):
                    idx, example = future_to_example[future]
                    try:
                        result = future.result()
                        results[idx] = (result, example)
                    except Exception as e:
                        self.logger.error(f"Future error for example {idx}: {str(e)}")
                        results[idx] = (
                            {
                                "task_id": example.get("task_id"),
                                "prompt": example.get("prompt", ""),
                                "entry_point": example.get("entry_point", ""),
                                "is_stdin": example.get("is_stdin", False),
                                "content": example["model_answer"],
                                "difficulty": example["difficulty"],
                                "correctness": False,
                                "reason": f"Future error: {str(e)}",
                                "test_input": None,
                                "test_output": None,
                                "test_expected": None,
                                "num_tests_passed": 0,
                                "num_tests_failed": 0,
                                "test_results": [],
                            },
                            example,
                        )

            all_repeat_results.append(results)

        final_metrics = {}

        # Compute pass@k metrics
        self.logger.info("Computing pass@k metrics...")

        results_by_task_id = defaultdict(list)
        results_by_task_id_and_difficulty = defaultdict(lambda: defaultdict(list))

        for repeat_results in all_repeat_results:
            for result, example in repeat_results:
                task_id = result["task_id"]
                difficulty = result["difficulty"]
                test_results = result.get("test_results", [])

                if test_results:
                    results_by_task_id[task_id].append(test_results)
                    results_by_task_id_and_difficulty[difficulty][task_id].append(test_results)
                else:
                    num_test_cases = len(example.get("test", [])) if "test" in example else 1
                    num_test_cases = max(num_test_cases, 1)
                    self.logger.debug(
                        f"Task {task_id} ({difficulty}): empty test_results, "
                        f"treating as all {num_test_cases} tests failed"
                    )
                    failed_results = [0] * num_test_cases
                    results_by_task_id[task_id].append(failed_results)
                    results_by_task_id_and_difficulty[difficulty][task_id].append(failed_results)

        k_list = [1]
        if self.n_repeat >= 10:
            k_list.append(5)
        if self.n_repeat >= 20:
            k_list.append(10)
        if self.n_repeat >= 32:
            k_list.append(16)
        if self.n_repeat >= 40:
            k_list.append(20)
        if self.n_repeat >= 64:
            k_list.append(32)

        self.logger.info(f"Computing pass@k metrics for k={k_list} (n_repeat={self.n_repeat})")

        if results_by_task_id:
            pass_at_k_overall = compute_metrics_from_results(dict(results_by_task_id), k_list=k_list)

            for k in k_list:
                key = f"pass@{k}"
                if key in pass_at_k_overall:
                    final_metrics[key] = pass_at_k_overall[key]
                    self.logger.info(f"Overall {key}: {pass_at_k_overall[key]:.2%}")

        for difficulty in results_by_task_id_and_difficulty:
            if difficulty in results_by_task_id_and_difficulty:
                diff_results = dict(results_by_task_id_and_difficulty[difficulty])
                if diff_results:
                    pass_at_k_diff = compute_metrics_from_results(diff_results, k_list=k_list)

                    for k in k_list:
                        key = f"pass@{k}"
                        if key in pass_at_k_diff:
                            final_metrics[f"pass@{k}_{difficulty}"] = pass_at_k_diff[key]
                            self.logger.info(f"{key} {difficulty}: {pass_at_k_diff[key]:.2%}")

        final_metrics["examples"] = [result for result, _ in results]
        final_metrics["num_total"] = num_questions
        final_metrics["num_repeat"] = self.n_repeat

        return final_metrics

    def load_questions(self):
        """Load LiveCodeBenchV6 questions from HuggingFace."""
        from collections import defaultdict as _defaultdict
        from datasets import Dataset, concatenate_datasets, load_dataset

        self.logger.info("Loading LiveCodeBenchV6 questions from livecodebench/code_generation_lite...")
        cpu_count = os.cpu_count()
        lcb_codegen = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
        self.logger.info(f"Loaded {len(lcb_codegen)} problems from livecodebench/code_generation_lite")
        ds = lcb_codegen.filter(filter_by_contest_date)
        self.logger.info(f"{len(ds)} problems after date filter (Feb-May 2025)")
        # Avoids "pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays" when mapping
        processed_shards = []
        num_shards = 4
        for i in range(num_shards):
            shard = ds.shard(num_shards=num_shards, index=i)
            shard = shard.map(
                lambda example: {"private_test_cases": translate_private_test_cases(example["private_test_cases"])},
                num_proc=cpu_count,
            )
            shard = shard.map(map_to_example, remove_columns=ds.column_names, load_from_cache_file=False)
            processed_shards.append(shard)
        ds = concatenate_datasets(processed_shards)
        return ds


# ---------------------------------------------------------------------------
# Public evaluator class
# ---------------------------------------------------------------------------


class LiveCodeBenchEvaluator(BaseEvaluator):
    """LiveCodeBench v6 evaluator ported from Apple ml-ssd."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_benchmark_name(self) -> str:
        return "livecodebench-v6"

    def evaluate(self, model_path: str, seed: int = 42) -> BenchmarkResult:
        """Evaluate model on LiveCodeBench v6 and return pass@k metrics.

        This runs in-process. For SLURM submission, use generate_eval_script() instead.
        """
        try:
            from transformers import AutoTokenizer
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "LCB evaluation requires vllm and transformers. "
                "Install with: pip install -e '.[ssd]'"
            ) from exc

        llm = LLM(model=model_path, tensor_parallel_size=self.config.get("tensor_parallel_size", 1))
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        sampling_params = self.config.get("sampling_params", {
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        })
        max_tokens = self.config.get("max_tokens", 32768)
        n_repeat = self.config.get("n_repeat", 20)

        benchmark = _LiveCodeBenchV6(
            llm=llm,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            n_repeat=n_repeat,
            sampling_params=sampling_params,
            seed=self.config.get("seed", [0, 1234, 1234, 1234]),
        )
        results = benchmark.run()

        metrics = {k: v for k, v in results.items() if isinstance(v, float) and k.startswith("pass@")}
        return BenchmarkResult(
            benchmark="livecodebench-v6",
            metrics=metrics,
            seed=seed,
            num_samples=results.get("num_total", 0),
            details=results,
        )


# ---------------------------------------------------------------------------
# Script generation for SLURM / standalone execution
# ---------------------------------------------------------------------------


def generate_eval_script(config: dict[str, Any]) -> str:
    """Generate a self-contained Python script for LCB v6 evaluation.

    The produced script can be submitted to SLURM or run standalone.
    It loads the model, evaluates on LiveCodeBench v6, and writes
    results to a JSON file.
    """
    model_path = config.get("model_path", "")
    output_path = config.get("output_path", "lcb_eval_results.json")
    tensor_parallel_size = config.get("tensor_parallel_size", 4)
    max_tokens = config.get("max_tokens", 32768)
    n_repeat = config.get("n_repeat", 20)
    sampling_params = config.get("sampling_params", {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
    })
    seed = config.get("seed", [0, 1234, 1234, 1234])

    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"LCB v6 Evaluation Script -- generated by auto-coder-trainer.\"\"\"

        import json
        import logging
        import sys
        from pathlib import Path

        from transformers import AutoTokenizer
        from vllm import LLM

        from trainers.ssd.lcb_evaluator import _LiveCodeBenchV6

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
        logger = logging.getLogger("lcb_eval")

        def main():
            model_path = {model_path!r}
            output_path = Path({output_path!r})
            output_path.parent.mkdir(parents=True, exist_ok=True)

            tensor_parallel_size = {tensor_parallel_size}
            max_tokens = {max_tokens}
            n_repeat = {n_repeat}
            sampling_params = {sampling_params!r}
            seed = {seed!r}

            logger.info("Loading model: %s", model_path)
            llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            benchmark = _LiveCodeBenchV6(
                llm=llm,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                n_repeat=n_repeat,
                sampling_params=sampling_params,
                seed=seed,
            )
            results = benchmark.run()

            # Separate serialisable metrics from non-serialisable details
            serialisable = {{
                k: v for k, v in results.items()
                if isinstance(v, (int, float, str, list, dict))
            }}

            with open(output_path, "w") as f:
                json.dump(serialisable, f, indent=2, default=str)

            logger.info("Results written to %s", output_path)

            # Print summary
            for k, v in results.items():
                if isinstance(v, float) and k.startswith("pass@"):
                    logger.info("  %s: %.2f%%", k, v * 100)

        if __name__ == "__main__":
            main()
    """)
