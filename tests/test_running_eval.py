import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import running_eval as ev

DATA = Path(__file__).resolve().parents[1] / 'scenarios' / 'running_analysis.json'


class RunningEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = ev.load_dataset(DATA)

    def answer(self, case):
        return dict(copy.deepcopy(case['expected']), explanation='合成数据说明')

    def trial(self, case, repeat=1):
        r = ev.grade(case, json.dumps(self.answer(case)))
        r.update(repeat=repeat, latency_s=0.1)
        return r

    def run_data(self):
        return {'schema': 'running-analysis-run/v1', 'execution_kind': 'fixture',
                'label': 'test', 'contract': {'dataset_sha256': 'a',
                'case_ids': [c['id'] for c in self.dataset['cases']],
                'settings': {'repeats_per_case': 1}},
                'summary': {'automated_contract_gate': 'PASS'},
                'results': [self.trial(c) for c in self.dataset['cases']]}

    def test_all_golden_answers_pass(self):
        for case in self.dataset['cases']:
            with self.subTest(case=case['id']):
                self.assertEqual(ev.grade(case, json.dumps(self.answer(case)))['status'], 'pass')

    def test_strict_bad_json_and_schema(self):
        case = self.dataset['cases'][0]
        good = self.answer(case)
        variants = ['{', '{"x": NaN}', '{"x":1,"x":2}', json.dumps({**good, 'activity_count': True}),
                    json.dumps({**good, 'activity_count': 1.0}), json.dumps({**good, 'needs_clarification': 1}),
                    json.dumps({**good, 'evidence_ids': ['a','a']}), json.dumps({**good, 'extra': 'bad'}),
                    json.dumps({k:v for k,v in good.items() if k != 'plan_status'})]
        for raw in variants:
            with self.subTest(raw=raw):
                self.assertEqual(ev.grade(case, raw)['status'], 'error')

    def test_request_error_unknown(self):
        case = self.dataset['cases'][0]
        result = ev.grade(case, json.dumps(self.answer(case)), 'network error')
        result['latency_s'] = 0.1
        self.assertEqual(result['status'], 'error')
        self.assertEqual(ev.summarize([result])['automated_contract_gate'], 'UNKNOWN')

    def test_critical_failure_blocks_even_with_high_pass_rate(self):
        case = next(c for c in self.dataset['cases'] if 'plan_status' in c['critical_fields'])
        answer = self.answer(case)
        answer['plan_status'] = 'active' if answer['plan_status'] != 'active' else 'paused'
        bad = ev.grade(case, json.dumps(answer))
        self.assertIn('plan_status', bad['critical_failures'])
        summary = ev.summarize([self.trial(case) for _ in range(99)] + [bad])
        self.assertEqual(summary['automated_contract_gate'], 'BLOCK')
        self.assertGreater(summary['pass_rate'], 0.98)

    def test_compare_refuses_hash_settings_mismatch(self):
        old = self.run_data()
        for key in ('dataset_sha256', 'settings'):
            new = copy.deepcopy(old)
            new['contract'][key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError):
                ev.compare_runs(old, new)

    def test_compare_refuses_missing_or_duplicate_trials(self):
        old = self.run_data()
        new = copy.deepcopy(old)
        new['results'].pop()
        with self.assertRaises(ValueError):
            ev.compare_runs(old, new)
        new = copy.deepcopy(old)
        new['results'].append(copy.deepcopy(new['results'][0]))
        with self.assertRaises(ValueError):
            ev.compare_runs(old, new)

    def test_compare_does_not_trust_summary(self):
        old = self.run_data()
        new = copy.deepcopy(old)
        case = self.dataset['cases'][0]
        answer = self.answer(case)
        answer['plan_status'] = 'paused' if answer['plan_status'] != 'paused' else 'active'
        changed = ev.grade(case, json.dumps(answer))
        changed.update(repeat=1, latency_s=0.1)
        new['results'][0] = changed
        self.assertEqual(ev.compare_runs(old, new)['automated_contract_gate'], 'BLOCK')

    def test_both_runs_missing_same_trial_is_not_pass(self):
        old = self.run_data()
        old['results'].pop()
        with self.assertRaises(ValueError):
            ev.compare_runs(old, copy.deepcopy(old))

    def test_empty_summary_unknown(self):
        self.assertEqual(ev.summarize([])['automated_contract_gate'], 'UNKNOWN')

    def test_forged_trial_status_is_rejected(self):
        old = self.run_data()
        new = copy.deepcopy(old)
        new['results'][0]['checks'][0]['passed'] = False
        with self.assertRaises(ValueError):
            ev.compare_runs(old, new)

    def test_expected_answers_not_in_request(self):
        case = self.dataset['cases'][0]
        config = {'target': {'base_url': 'http://localhost:1/v1', 'model': 'x'},
                  'evaluation': {'repeats_per_case': 1, 'max_tokens': 10, 'temperature': 0, 'timeout_seconds': 1}}
        with patch.object(ev, 'request_completion', return_value=(json.dumps(self.answer(case)), 0.1, None)) as request:
            ev.execute(config, {'version': 'test', 'cases': [case]})
        messages = request.call_args.args[2]
        payload = json.loads(messages[-1]['content'])
        self.assertEqual(set(payload), {'case_id', 'input'})
        self.assertNotIn('expected', payload)

    def test_execute_does_not_save_api_key(self):
        secret = 'secret-never-save-in-test'
        config = {'target': {'base_url': 'http://127.0.0.1:1234/v1', 'model': 'x', 'api_key': secret},
                  'evaluation': {'repeats_per_case': 1, 'max_tokens': 10, 'temperature': 0, 'timeout_seconds': 1}}
        case = self.dataset['cases'][0]
        with patch.object(ev, 'request_completion', return_value=(json.dumps(self.answer(case)), 0.1, None)):
            result = ev.execute(config, {'version': 'test', 'cases': [case]})
        self.assertNotIn(secret, json.dumps(result))

    def test_loopback_demo_and_cli_exit(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            comparison = ev.demo(d)
            self.assertEqual(comparison['automated_contract_gate'], 'BLOCK')
            self.assertEqual(comparison['baseline_summary']['automated_contract_gate'], 'PASS')
            self.assertTrue((Path(d) / 'comparison.md').exists())
            args = ['compare', '--baseline', str(Path(d)/'fixture-baseline.json'),
                    '--candidate', str(Path(d)/'fixture-regressed.json'),
                    '--report', str(Path(d)/'cli.md'), '--fail-on-regression']
            self.assertEqual(ev.main(args), 1)
            self.assertIn('模拟演示', (Path(d)/'cli.md').read_text())

    def test_compare_output_cannot_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            old = Path(d) / 'baseline.json'
            new = Path(d) / 'candidate.json'
            ev.write_json(old, self.run_data())
            ev.write_json(new, self.run_data())
            original = old.read_bytes()
            for report in (old, Path(d) / 'baseline.md', Path(d) / 'candidate.md'):
                self.assertEqual(ev.main(['compare', '--baseline', str(old), '--candidate', str(new),
                                          '--report', str(report)]), 2)
            self.assertEqual(old.read_bytes(), original)

    def test_run_unknown_is_nonzero_and_preserves_result(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            config_path = Path(d) / 'config.yaml'
            config_path.write_text('target:\n  base_url: http://127.0.0.1:1/v1\n  model: x\nevaluation:\n  repeats_per_case: 1\n')
            output = Path(d) / 'run.json'
            with patch.object(ev, 'request_completion', return_value=('', 0.1, 'HTTP 503')):
                self.assertEqual(ev.main(['run', '--config', str(config_path), '--output', str(output)]), 2)
            self.assertEqual(json.loads(output.read_text())['summary']['automated_contract_gate'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
