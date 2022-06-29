import os
import logging
import pickle
import random
import sys
import ujson
import argparse
import uvicorn
import subprocess
from typing import List, NoReturn, Tuple, Dict, Optional, TypeVar
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from featurize import Vocab, collect_features_as_list, counter_vectorize
from recommend_algo.recommend import print_similar_and_completions, fail
from elastic_api import ES


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--corpus",
        action="store",
        dest="corpus",
        help="Process raw ASTs, featurize, and store in the working directory.",
        required=False,
    )
    parser.add_argument(
        "-d",
        "--working-dir",
        action="store",
        dest="working_dir",
        default="./tmpout",
        help="Working directory.",
        required=False,
    )
    parser.add_argument(
        "-f",
        "--file-query",
        action="append",
        dest="file_query",
        default=[],
        help="Files containing the query code.",
    )
    parser.add_argument(
        "-i",
        "--index-query",
        type=int,
        action="store",
        dest="index_query",
        default=None,
        help="Index of the query AST in the corpus.",
    )
    parser.add_argument(
        "-t",
        "--testall",
        dest="testall",
        action="store_true",
        default=False,
        help="Sample config.N_SAMPLES snippets and search.",
    )
    parser.add_argument(
        "-r",
        "--run",
        dest="rest_service",
        action="store_true",
        default=False,
        help="Start the RESTful service.",
    )
    options = parser.parse_args()
    logging.info(options)
    return options


# 从records中随机选出n个record（删减过的）
def sample_n_records(n, total_len):
    ret_indices = []
    ret_records = []
    random.seed(config.SEED)
    for j in range(10000):
        if len(ret_indices) < n:
            i = random.randint(0, total_len - 1)
            if not (i in ret_indices):
                record = get_record_part(get_record(i))
                if record != None:
                    ret_indices.append(i)
                    ret_records.append(record)
        else:
            logging.info("Sampled {len} records".format(len=len(ret_indices)))
            return (ret_indices, ret_records)
    logging.info("Sampled {len} records".format(len=len(ret_indices)))
    return (ret_indices, ret_records)


def get_sub_ast_aux(ast, beginline, n_lines, stop=False):
    if isinstance(ast, list):
        if stop:
            return (stop, None)
        else:
            ret = []
            for elem in ast:
                (stop, tmp) = get_sub_ast_aux(elem, beginline, n_lines, stop)
                if tmp != None:
                    ret.append(tmp)
            if len(ret) >= 2:
                return (stop, ret)
            else:
                return (True, None)
    elif isinstance(ast, dict) and "label" not in ast:
        if (
                #"leaf" not in ast
                # or not ast["leaf"]
                (not stop and ast["line"] - beginline <= n_lines * 0.6)
        ):
            return (stop, ast)
        else:
            return (True, None)
    else:
        return (stop, ast)


def copy_record_with_ast(record, ast):
    ret = dict(record)
    ret["ast"] = ast
    return ret


# 获取record的删减版，用于测试
def get_record_part(record):
    n_lines = record["endline"] - record["beginline"] + 1
    if n_lines < config.SAMPLE_METHOD_MIN_LINES:
        return None
    else:
        (_, ast) = get_sub_ast_aux(record["ast"], record["beginline"], n_lines)
        if ast == None:
            return None
        else:
            ret = copy_record_with_ast(record, ast)
            ret["features"] = collect_features_as_list(
                ast, False, False, vocab)
            if len(ret['features']) == 0:
                return None
            return ret


# 从records库拿出第idx个record，将其删减后输入，进行测试，看能否得到原record
def test_record_at_index(idx):
    record = get_record(idx)
    record_part = get_record_part(record)
    if record_part != None:
        print_similar_and_completions(
            record_part, get_records, vectorizer, counter_matrix)


def featurize_and_test_record(record: dict) -> List[str]:
    record["features"] = collect_features_as_list(
        record["ast"], False, False, vocab)
    record["index"] = -1
    if len(record["features"]) > 0:
        return print_similar_and_completions(record, get_records, vectorizer, counter_matrix)


def test_all(total_len):
    N = config.N_SAMPLES
    (sampled_indices, sampled_records) = sample_n_records(N, total_len)
    for k, record in enumerate(sampled_records):
        print(f"{k}: ", end="")
        print_similar_and_completions(
            record, get_records, vectorizer, counter_matrix)
    print(
        f"# of sampled method: {len(sampled_records)}, # of Failed to match original method: {fail}")


def load_matrix(counter_path):
    with open(counter_path, "rb") as outf:
        (vectorizer, counter_matrix) = pickle.load(outf)
        logging.info("Read vectorizer and counter matrix.")
    return (vectorizer, counter_matrix)


def setup(ast_path):
    global vectorizer
    global counter_matrix
    global vocab
    os.makedirs(options.working_dir, exist_ok=True)
    if ast_path is not None:
        es_instance.set_mapping()
        vocab = Vocab.load(options.working_dir, init=True)
        run_buildIndex(ast_path, options.working_dir)
    else:
        vocab = Vocab.load(options.working_dir, init=False)
    config.g_vocab = vocab
    config.RECORD_QUANTITY = get_recordQuantity()
    (vectorizer, counter_matrix) = load_matrix(
        os.path.join(options.working_dir, config.TFIDF_FILE)
    )


def get_record(id: int):
    return es_instance.get_with_id(id)


def get_records(ids: List[int]):
    return es_instance.mget_records_with_ids(ids)


def insert_record(id: int, record: dict):
    return es_instance.insert_with_id(id, record)


def get_nextId():
    return 0
    # return es_instance.records_count()


def get_recordQuantity():
    return es_instance.records_count()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins='http://localhost:8060',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/test")
def run_pipeline(language: str, path: str, line: int) -> dict:
    if language.lower() == 'java':
        command = 'java -jar reference/target/ANTLR4SimpleAST-1.0-SNAPSHOT-jar-with-dependencies.jar compilationUnit stdout {inputpath}'.format(
            inputpath=path)
    if language.lower() == 'python':
        return
    complete = subprocess.run(
        command, stdout=subprocess.PIPE, check=True, shell=True)
    records = complete.stdout.strip().split('\n')
    for record in records:
        record = ujson.loads(record)
        if record['beginline'] <= line and line <= record['endline']:
            results = featurize_and_test_record(record)
            break
    else:
        results = ['// 请将光标移至方法所在行。']
    return {"recommendation": results}


def run_buildIndex(path: str, working_dir: str, language: str = 'python') -> NoReturn:
    filecnt = 0
    id = get_nextId()

    def get_src_path(path: str, output_file: str, language: str) -> NoReturn:
        suffix = '.java' if language == 'java' else '.py'
        if os.path.exists(path):
            for wholepath in [os.path.join(path, f) for f in os.listdir(path)]:
                if os.path.isdir(wholepath):
                    get_src_path(wholepath, output_file, language)
                elif os.path.isfile(wholepath):
                    if wholepath.endswith(suffix) and 'test' not in wholepath.lower():
                        relpath = os.path.relpath(wholepath)
                        output_file.write(relpath)
                        output_file.write('\n')
                        nonlocal filecnt
                        filecnt = filecnt + 1
                        logging.info("file quantity: " +
                                     str(filecnt) + ', ' + relpath)

    def convert2Ast(language: str):
        language = language.lower()
        inputpath = os.path.join(working_dir, "{0}_path.txt".format(language))
        outputpath = os.path.join(working_dir, "{0}_ast.json".format(language))
        if os.path.exists(inputpath):
            os.remove(inputpath)
        if os.path.exists(outputpath):
            os.remove(outputpath)
        with open(inputpath, "w") as f:
            get_src_path(path, f, language)
        if language == 'java':
            command = 'java -jar ast_converter/ANTLR4SimpleAST-1.0-SNAPSHOT-jar-with-dependencies.jar compilationUnit {outputpath} {inputpath}'.format(
                outputpath=outputpath, inputpath=inputpath)
            subprocess.run(command, cwd=os.getcwd(), check=True, shell=True)
        if language == 'python':
            from ast_converter.py_converter import execute as py2ast_execute
            with open(inputpath, 'r') as f:
                for line in f:
                    py2ast_execute(line.strip(), outputpath)

    def run_featurize(id: int):
        with open(os.path.join(working_dir, "{0}_ast.json".format(language)), "r") as f:
            for line in f:
                record = ujson.loads(line)
                record["features"] = collect_features_as_list(
                    record["ast"], True, False, vocab)
                record["index"] = id
                yield {
                    '_op_type': 'create',
                    '_index': '{0}_method_records'.format(language),
                    '_id': id,
                    '_source': record
                }
                id += 1
                logging.info("Has featurized: " + str(id))

    convert2Ast(language)
    # 批量插入（将generator传入，内部是并行异步的）
    es_instance.bulk(run_featurize(id))
    es_instance.refresh()
    vocab.dump(working_dir)
    logging.info("Dumped feature vocab.")
    print(get_recordQuantity())
    config.RECORD_QUANTITY = get_recordQuantity()
    assert config.RECORD_QUANTITY > 0
    counter_vectorize(
        get_records,
        os.path.join(options.working_dir, config.TFIDF_FILE),
    )


def collect_features_as_list_wapperForCpp(ast, is_init: bool, is_counter: bool):
    return collect_features_as_list(ast, is_init, is_counter, config.g_vocab)


if __name__ == "__main__":
    options = parse_args()
    vectorizer = None
    counter_matrix = None
    vocab = None
    es_instance = ES('python')
    setup(options.corpus)
    if options.index_query is not None:
        test_record_at_index(options.index_query)
    elif len(options.file_query) > 0:
        with open(options.file_query, "r") as f:
            for line in f:
                obj = ujson.loads(line)
                featurize_and_test_record(obj)
    elif options.testall:
        file = open('out.txt', 'w')
        sys.stdout = file
        config.TEST_ALL = True
        test_all(total_len=config.RECORD_QUANTITY)
        file.close()
    elif options.rest_service:
        uvicorn.run(app, host="127.0.0.1", port=8000)
