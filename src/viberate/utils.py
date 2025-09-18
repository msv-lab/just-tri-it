import sys
import shutil
import mistletoe
import json
from pathlib import Path

from viberate.cached_llm import Independent


def print_hr():
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    print('-' * width, file=sys.stderr)


def print_annotated_hr(message):
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    msg = f' {message} '
    dash_count = width - len(msg)
    if dash_count < 0:
        print(message)
        return
    left_dashes = dash_count // 2
    right_dashes = dash_count - left_dashes
    line = '-' * left_dashes + msg + '-' * right_dashes
    print(line, file=sys.stderr)


def panic(msg):
    print(msg, file=sys.stderr)
    exit(1)


class DataExtractionFailure(Exception):
    "Raised when failed to parse LLM output"
    pass    
    

def extract_code(content):
    """Extract first markdown code block"""
    parsed = mistletoe.Document(content)
    for child in parsed.children:
        if child.__class__.__name__ == "CodeFence":
            return child.children[0].content
    raise DataExtractionFailure


def extract_answer(s):
    if "<answer>" in s and "</answer>" in s and \
       s.index("<answer>") < s.index("</answer>"):
        return s.split("<answer>", 1)[1].split("</answer>", 1)[0]
    else:
        raise DataExtractionFailure


def gen_and_extract_answer_with_retry(model, prompt, num_retry=3):
    ind_model = Independent(model)
    ans = None
    for attempt in range(num_retry):
        try:
            ans = extract_answer(next(ind_model.sample(prompt, num_retry)))
            break
        except Exception as e:
            if attempt == num_retry - 1:
                raise Exception(f"did not get good response: {e}")
        pass
    return ans


class CompactJSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，将列表压缩到一行"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_indent = 0
        self.current_indent_str = ""
    
    def encode(self, obj):
        if isinstance(obj, list):
            # 对列表进行特殊处理，保持在一行
            return "[" + ", ".join(json.dumps(item, ensure_ascii=False) for item in obj) + "]"
        elif isinstance(obj, dict):
            # 对字典保持原有缩进格式
            return super().encode(obj)
        else:
            return json.dumps(obj, ensure_ascii=False)
    
    def iterencode(self, obj, _one_shot=False):
        # 重写 iterencode 方法以处理嵌套结构
        if isinstance(obj, list):
            yield "["
            first = True
            for item in obj:
                if first:
                    first = False
                else:
                    yield ", "
                yield from self.iterencode(item)
            yield "]"
        elif isinstance(obj, dict):
            yield "{\n"
            self.current_indent += self.indent
            self.current_indent_str = " " * self.current_indent
            first = True
            for key, value in obj.items():
                if first:
                    first = False
                else:
                    yield ",\n"
                yield self.current_indent_str + json.dumps(key, ensure_ascii=False) + ": "
                yield from self.iterencode(value)
            self.current_indent -= self.indent
            self.current_indent_str = " " * self.current_indent
            yield "\n" + self.current_indent_str + "}"
        else:
            yield json.dumps(obj, ensure_ascii=False)

def write_dict_to_json(data_dict, file_path, indent=4, ensure_ascii=False):
    """将字典写入 JSON 文件，列表保持在一行"""
    file_path = Path(file_path)
    
    try:
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用自定义编码器写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            if any(isinstance(v, list) for v in data_dict.values()):
                # 如果数据中有列表，使用自定义编码器
                json.dump(data_dict, f, indent=indent, ensure_ascii=ensure_ascii, 
                         cls=CompactJSONEncoder)
            else:
                # 如果没有列表，使用标准编码器
                json.dump(data_dict, f, indent=indent, ensure_ascii=ensure_ascii)
        
        print(f"数据已成功写入: {file_path}")
        return True
        
    except Exception as e:
        print(f"写入JSON文件时出错: {e}")
        return False
