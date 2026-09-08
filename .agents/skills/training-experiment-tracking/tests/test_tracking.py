"""在临时 Git 仓库验证索引行为，不访问云资源。"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/tracking.py'
spec = importlib.util.spec_from_file_location('tracking', SCRIPT)
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Test')
        policy = self.repo / tracking.POLICY
        policy.parent.mkdir(parents=True)
        policy.write_text(json.dumps({'schema_version': 1, 'allowed_dirty_paths': ['records/index.json', 'results.md']}))
        (self.repo / 'train.py').write_text('original\n')
        (self.repo / 'results.md').write_text('initial\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'base')
        self.index = self.repo / 'records/index.json'

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args]).decode().strip()

    def cli(self, *args, expected=0):
        proc = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, expected, proc.stdout + proc.stderr)
        return json.loads(proc.stdout)

    def write_input(self, row, name='input.json'):
        path = self.root / name
        path.write_text(json.dumps(row))
        return path

    def row(self, run='r1', job='j1', source=None):
        return {'experiment_id': 'e1', 'job_id': job, 'run_id': run, 'git_commit': 'a' * 40,
                'status': 'Succeeded' if run else 'Failed', 'resume_from': source,
                'training_date': '2026-09-07T12:20:08+08:00',
                'recorded_date': '2026-09-08T09:30:00+08:00', 'output_uri': 'oss://bucket/runs/' + (run or job)}

    def record(self, row, expected=0, replace=False):
        args = ['record-result', '--index', self.index, '--input', self.write_input(row)]
        if replace:
            args.append('--replace')
        return self.cli(*args, expected=expected)

    def test_time_and_output_fields_are_required(self):
        for field in ('training_date', 'recorded_date', 'output_uri'):
            with self.subTest(field=field):
                row = self.row()
                del row[field]
                result = self.record(row, expected=2)
                self.assertIn(field, result['error'])
                self.assertFalse(self.index.exists())
        self.record({**self.row(None), 'training_date': None, 'recorded_date': None, 'output_uri': None})

    def test_invalid_time_or_output_preserves_index(self):
        self.record(self.row())
        before = self.index.read_bytes()
        for field in ('training_date', 'recorded_date', 'output_uri'):
            for value in (True, 123, '', '   ', []):
                with self.subTest(field=field, value=value):
                    self.record({**self.row(), field: value}, expected=2, replace=True)
                    self.assertEqual(before, self.index.read_bytes())

    def test_workspace_policy(self):
        self.assertEqual(self.cli('check-workspace', '--repo', self.repo)['state'], 'clean')
        (self.repo / 'results.md').write_text('new result')
        report = self.cli('check-workspace', '--repo', self.repo)
        self.assertEqual(report['state'], 'allowed_dirty')
        self.assertTrue(report['dirty'])
        self.assertEqual(report['changes'], [])
        policy_path = self.repo / tracking.POLICY
        policy = json.loads(policy_path.read_text())
        policy['allowed_dirty_paths'].extend(['train.py', tracking.POLICY])
        policy_path.write_text(json.dumps(policy))
        (self.repo / 'train.py').write_text('new source')
        self.git('add', '.')
        report = self.cli('check-workspace', '--repo', self.repo)
        self.assertEqual(report['state'], 'allowed_dirty')
        self.assertEqual(report['changes'], [])

    def test_skill_directory_includes_policy(self):
        policy_path = self.repo / tracking.POLICY
        policy = json.loads(policy_path.read_text())
        policy['allowed_dirty_paths'].append('.agents/skills/')
        policy_path.write_text(json.dumps(policy))
        self.git('add', '.')
        self.git('commit', '-qm', 'allow skills')
        skill = self.repo / '.agents/skills/example/SKILL.md'
        skill.parent.mkdir()
        skill.write_text('draft')
        result = self.cli('check-workspace', '--repo', self.repo)
        self.assertEqual(result['state'], 'allowed_dirty')
        self.assertTrue(result['dirty'])
        self.assertEqual(result['changes'], [])
        policy_path.write_text(json.dumps(policy) + '\n')
        self.git('add', str(policy_path))
        result = self.cli('check-workspace', '--repo', self.repo)
        self.assertEqual(result['state'], 'allowed_dirty')
        self.assertEqual(result['changes'], [])

    def test_directory_boundary_and_rename_source(self):
        policy_path = self.repo / tracking.POLICY
        policy = json.loads(policy_path.read_text())
        policy['allowed_dirty_paths'].append('.agents/skills/')
        policy_path.write_text(json.dumps(policy))
        self.git('add', '.')
        self.git('commit', '-qm', 'allow skills')
        sibling = self.repo / '.agents/skills-extra/file.md'
        sibling.parent.mkdir()
        sibling.write_text('outside')
        self.cli('check-workspace', '--repo', self.repo, expected=2)
        sibling.unlink()
        self.git('mv', 'train.py', '.agents/skills/moved.py')
        result = self.cli('check-workspace', '--repo', self.repo, expected=2)
        moved = next(c for c in result['changes'] if 'train.py' in c['paths'])
        self.assertFalse(moved['allowed'])

    def test_missing_policy_is_blocked(self):
        self.git('rm', tracking.POLICY)
        self.git('commit', '-qm', 'remove policy')
        result = self.cli('check-workspace', '--repo', self.repo, expected=2)
        self.assertTrue(result['errors'])
        self.assertEqual(set(result), {'state', 'errors'})

    def test_conflict_is_not_allowed(self):
        branch = self.git('branch', '--show-current')
        self.git('checkout', '-qb', 'other')
        (self.repo / 'results.md').write_text('other\n')
        self.git('commit', '-qam', 'other')
        self.git('checkout', '-q', branch)
        (self.repo / 'results.md').write_text('main\n')
        self.git('commit', '-qam', 'main')
        subprocess.run(['git', '-C', str(self.repo), 'merge', 'other'], capture_output=True)
        result = self.cli('check-workspace', '--repo', self.repo, expected=2)
        self.assertTrue(any(c['conflict'] for c in result['changes']))

    def test_failed_result_and_multiple_runs_on_dirty_repo(self):
        (self.repo / 'train.py').write_text('dirty')
        self.record(self.row(None, 'failed-job'))
        self.record(self.row('r1'))
        self.record(self.row('r2'))
        records = tracking.read_json(self.index)['experiments']['e1']['runs']
        self.assertEqual(set(records), {'job:failed-job', 'run:r1', 'run:r2'})
        self.assertEqual(records['run:r1']['job_id'], records['run:r2']['job_id'])

    def test_idempotence_and_explicit_replacement(self):
        row = self.row()
        self.record(row)
        before = self.index.read_bytes()
        os.utime(self.index, ns=(1_000_000_000, 1_000_000_000))
        before_stat = self.index.stat()
        self.record(row)
        self.assertEqual(before_stat.st_mtime_ns, self.index.stat().st_mtime_ns)
        self.assertEqual(before_stat.st_ino, self.index.stat().st_ino)
        self.assertEqual(before, self.index.read_bytes())
        changed = {**row, 'metrics': {'accuracy': 0.8}}
        self.record(changed, expected=2)
        self.assertEqual(before, self.index.read_bytes())
        self.record(changed, replace=True)
        self.assertEqual(tracking.read_json(self.index)['experiments']['e1']['runs']['run:r1'], {k: v for k, v in changed.items() if k != 'experiment_id'})
        self.record({**changed, 'job_id': 'another'}, expected=2, replace=True)

    def test_resume_and_cycles(self):
        self.record(self.row())
        source = {'run_id': 'r1', 'checkpoint_uri': 'oss://r1/last.pt'}
        self.record(self.row('r2', 'j2', source))
        self.record(self.row('r3', 'j3', source))
        before = self.index.read_bytes()
        self.record(self.row('r1', 'j1', {'run_id': 'r2', 'checkpoint_uri': 'oss://r2/last.pt'}),
                    expected=2, replace=True)
        self.assertEqual(before, self.index.read_bytes())
        self.record(self.row('r4', 'j4', {'run_id': 'absent', 'checkpoint_uri': 'oss://x'}), expected=2)
        self.assertEqual(before, self.index.read_bytes())
        self.record(self.row('r4', 'j4', {'run_id': None, 'external': True, 'checkpoint_uri': 'oss://x'}))

    def test_experiments_group_runs_and_preserve_versions(self):
        self.record(self.row('r1'))
        self.record({**self.row('r2', 'j2'), 'git_commit': 'b' * 40})
        self.record({**self.row('r3', 'j3'), 'experiment_id': 'e2'})
        index = tracking.read_json(self.index)
        self.assertEqual(set(index['experiments']), {'e1', 'e2'})
        runs = index['experiments']['e1']['runs']
        self.assertEqual(set(runs), {'run:r1', 'run:r2'})
        self.assertNotEqual(runs['run:r1']['git_commit'], runs['run:r2']['git_commit'])
        self.assertNotIn('experiment_id', runs['run:r1'])
        before = self.index.read_bytes()
        self.record({**self.row('r1'), 'experiment_id': 'e2'}, expected=2, replace=True)
        self.assertEqual(before, self.index.read_bytes())

    def test_invalid_and_old_index_preserved(self):
        self.record({**self.row(), 'experiment_id': True}, expected=2)
        self.assertFalse(self.index.exists())
        self.index.parent.mkdir()
        self.index.write_text('{broken')
        self.record(self.row(), expected=2)
        self.assertEqual(self.index.read_text(), '{broken')
        self.index.write_text('{"schema_version": 1, "attempts": {}}')
        self.record(self.row(), expected=2)
        self.assertEqual(self.index.read_text(), '{"schema_version": 1, "attempts": {}}')

    def test_concurrent_records(self):
        processes = []
        for run in ('r1', 'r2'):
            payload = self.write_input(self.row(run), run + '.json')
            processes.append(subprocess.Popen([sys.executable, str(SCRIPT), 'record-result',
                '--index', str(self.index), '--input', str(payload)], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True))
        for proc in processes:
            out, err = proc.communicate(timeout=20)
            self.assertEqual(proc.returncode, 0, out + err)
        self.assertEqual(set(tracking.read_json(self.index)['experiments']['e1']['runs']), {'run:r1', 'run:r2'})


if __name__ == '__main__':
    unittest.main()
