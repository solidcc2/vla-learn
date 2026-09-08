#!/usr/bin/env python3
"""本地实验索引工具；仅使用 Python 标准库。"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

POLICY = '.agents/skills/training-experiment-tracking/config.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'JSON 键重复：{key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def git(repo, *args):
    proc = subprocess.run(['git', '-C', str(repo), *args], capture_output=True)
    require(proc.returncode == 0, f'Git 查询失败：{args[0]}')
    return proc.stdout


def check_workspace(repo):
    root = Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip())
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    try:
        policy = read_json(root / POLICY)
        require(isinstance(policy, dict) and policy.get('schema_version') == 1, '白名单 schema_version 必须为 1')
        allowed = policy['allowed_dirty_paths']
        require(isinstance(allowed, list), '白名单必须为路径列表')
        for path in allowed:
            require(isinstance(path, str) and path and not path.startswith('/')
                    and all(part not in ('', '.', '..') for part in path.removesuffix('/').split('/'))
                    and not any(c in path for c in '*?[]'),
                    '白名单使用仓库相对路径，目录以 / 结尾')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {'state': 'blocked', 'errors': [f'工作区中的 {POLICY} 不可用：{exc}']}
    tokens = git(root, 'status', '--porcelain=v1', '-z', '--untracked-files=all',
                 '--ignore-submodules=none').split(b'\0')
    changes = []
    i = 0
    while i < len(tokens) and tokens[i]:
        token = os.fsdecode(tokens[i])
        status, path = token[:2], token[3:]
        paths = [path]
        if 'R' in status or 'C' in status:
            i += 1
            paths.append(os.fsdecode(tokens[i]))
        conflict = 'U' in status or status in ('AA', 'DD')
        is_allowed = not conflict and all(
            any(p.startswith(entry) if entry.endswith('/') else p == entry
                for entry in allowed)
            for p in paths
        )
        changes.append({'status': status, 'paths': paths,
                        'conflict': conflict, 'allowed': is_allowed})
        i += 1
    state = ('blocked' if any(not c['allowed'] for c in changes)
             else 'allowed_dirty' if changes else 'clean')
    return {'state': state, 'head': head, 'dirty': bool(changes),
            'changes': [c for c in changes if not c['allowed']]}


@contextmanager
def locked(path):
    # 锁放在本机临时目录，避免锁文件自身使记录目录 dirty。
    name = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    lock_dir = Path(tempfile.gettempdir()) / f'training-tracking-{os.getuid()}'
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    with (lock_dir / name).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def save_index(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def is_nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def result_key(row):
    return 'run:' + row['run_id'] if row['run_id'] else 'job:' + row['job_id']


def check_record(row):
    require(isinstance(row, dict), '记录应为对象')
    for field in ('job_id', 'status', 'git_commit'):
        require(is_nonempty_string(row.get(field)), f'缺少有效 {field}')
    require('run_id' in row and (row['run_id'] is None or is_nonempty_string(row['run_id'])),
            'run_id 应为非空字符串，无训练进程时填 null')
    for field in ('training_date', 'recorded_date', 'output_uri'):
        require(field in row and (row[field] is None or is_nonempty_string(row[field])),
                f'{field} 必须提供，填非空字符串或 null')


def check_records(index):
    require(isinstance(index, dict) and index.get('schema_version') == 3
            and isinstance(index.get('experiments'), dict),
            '索引格式不符（需要 schema_version=3/experiments）；请排查，原文件未修改')
    rows = {}
    for experiment_id, experiment in index['experiments'].items():
        require(is_nonempty_string(experiment_id) and isinstance(experiment, dict)
                and isinstance(experiment.get('runs'), dict), '实验分组需要有效名称和 runs 对象')
        for key, row in experiment['runs'].items():
            require(key not in rows, f'记录在多个实验中重复：{key}')
            rows[key] = row
    for key, row in rows.items():
        check_record(row)
        require(key == result_key(row), f'{key} 与记录标识不一致')
        source = row.get('resume_from')
        if source is not None:
            require(isinstance(source, dict) and is_nonempty_string(source.get('checkpoint_uri')),
                    f'{key} 恢复来源缺少 checkpoint_uri')
            parent = source.get('run_id')
            require(parent is None or is_nonempty_string(parent), f'{key} 恢复来源 run_id 无效')
            if parent is None:
                require(source.get('external') is True, f'{key} 外部来源需 external=true')
            else:
                require('run:' + parent in rows, f'{key} 来源 run 尚未登记：{parent}')
            sha = source.get('sha256')
            require(sha is None or isinstance(sha, str) and len(sha) == 64
                    and all(c in '0123456789abcdef' for c in sha), f'{key} 恢复摘要格式无效')
    for key in rows:
        seen, current = set(), key
        while current is not None:
            require(current not in seen, f'{key} 恢复关系成环；请排查，原文件未修改')
            seen.add(current)
            source = rows[current].get('resume_from')
            current = 'run:' + source['run_id'] if source and source.get('run_id') else None


def record_result(args):
    path = Path(args.index)
    require('://' not in args.index and not path.is_symlink(), '需要本地文件系统索引路径')
    row = read_json(args.input)
    check_record(row)
    require(is_nonempty_string(row.get('experiment_id')), '需要 experiment_id')
    experiment_id = row.pop('experiment_id')
    key = result_key(row)
    result = {'saved': True, 'experiment_id': experiment_id, 'key': key}
    with locked(path):
        index = read_json(path) if path.exists() else {'schema_version': 3, 'experiments': {}}
        check_records(index)
        runs = index['experiments'].setdefault(experiment_id, {'runs': {}})['runs']
        old = runs.get(key)
        if old == row:
            return result
        require(old is None or args.replace,
                f'{key} 已有不同内容；核对后用 --replace 显式替换完整记录')
        if old is not None:
            require(old['job_id'] == row['job_id'],
                    '不能把已有 run 改归另一任务或实验')
        runs[key] = row
        check_records(index)
        save_index(path, index)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='检查 Git 工作区，或将单条 JSON 记录写入实验索引。',
        epilog='用 <子命令> --help 查看参数和示例。输出为 JSON；退出码 0 成功，2 阻断或错误。')
    commands = parser.add_subparsers(dest='command', required=True, title='子命令')
    check = commands.add_parser('check-workspace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='按白名单检查 Git 工作区变更',
        description='检查 --repo 所在的整个仓库，返回状态和阻断项，隐藏白名单内变更。',
        epilog='''示例：
  python tracking.py check-workspace --repo .

白名单：工作区中的 .agents/skills/training-experiment-tracking/config.json，修改立即参与匹配。
文件按仓库相对路径精确匹配；以 / 结尾的目录包含其下全部文件。
配置文件自身也按白名单匹配；合并冲突始终阻断，Git 已忽略文件不参与检查。

状态与退出码：
  clean          无变更，0
  allowed_dirty  仅白名单内变更，0
  blocked        其他变更、合并冲突或配置不可用，2''')
    check.add_argument('--repo', default='.', help='仓库内路径（默认：当前目录）')
    record = commands.add_parser('record-result', formatter_class=argparse.RawDescriptionHelpFormatter,
        help='新增或替换一条实验运行记录',
        description='从 --input 读取单条记录，写入 --index 指定的索引文件。',
        epilog='''示例：
  python tracking.py record-result --index experiments/index.json --input result.json

result.json：
{
  "experiment_id": "exp-A",
  "job_id": "job-A",
  "run_id": "run-A",
  "status": "Succeeded",
  "git_commit": "0123456789abcdef0123456789abcdef01234567",
  "training_date": "2026-09-07T12:20:08+08:00",
  "recorded_date": "2026-09-08T09:30:00+08:00",
  "output_uri": "oss://bucket/project/runs/run-A/",
  "resume_from": null
}

示例中除 resume_from 外均为必填字段。
run_id、training_date、recorded_date、output_uri 可填 null；时间不会自动生成。

续训时填写 resume_from，例如：
  {"run_id":"run-A","checkpoint_uri":"oss://bucket/runs/run-A/checkpoint.pt","sha256":null}
来源 run 需已登记；外部来源使用 run_id=null、external=true。

相同内容重复登记不变；不同内容需要 --replace，省略字段会被删除。
成功返回 saved、experiment_id、key；错误返回 error，退出码 2，原索引不变。''')
    record.add_argument('--index', required=True, help='本地索引 JSON 路径，不存在时创建')
    record.add_argument('--input', required=True, help='单条记录 JSON 文件，格式见下方示例')
    record.add_argument('--replace', action='store_true', help='以完整输入替换已有记录')
    args = parser.parse_args()
    try:
        if args.command == 'check-workspace':
            result = check_workspace(args.repo)
            code = 2 if result['state'] == 'blocked' else 0
        else:
            result, code = record_result(args), 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result, code = {'error': str(exc), 'saved': False}, 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
