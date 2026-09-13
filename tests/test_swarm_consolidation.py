import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('consolidation', Path(__file__).resolve().parents[1] / 'scripts/consolidate_swarm.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConsolidationTest(unittest.TestCase):
    def setUp(self):
        service = {'image': module.IMAGE.split('@')[0], 'environment': {'NVIDIA_VISIBLE_DEVICES': '1', 'ENABLE_GRADIO_UI': 'true'},
                   'deploy': {'replicas': 1, 'labels': {'caddy': 'http://index-api'}}, 'volumes': ['/models:/models']}
        self.before = {'services': {name: copy.deepcopy(service) for name in ('api-1', 'api-2', 'api-3')}}
        self.desired = {'services': {'api-3': copy.deepcopy(service)}}
        self.desired['services']['api-3']['image'] = module.IMAGE
        self.desired['services']['api-3']['deploy']['labels'] = {'caddy_0': 'http://index-api', 'caddy_0.reverse_proxy': 'index-tts_api-3.:8000'}

    def test_transition_keeps_old_models_running_until_route_verification(self):
        result = module.build_transition(self.before, self.desired)
        self.assertEqual(set(result['services']), {'api-1', 'api-2', 'api-3'})
        for name in ('api-1', 'api-2'):
            self.assertEqual(result['services'][name]['deploy']['replicas'], 1)
            self.assertEqual(result['services'][name]['deploy']['labels'], {})
            self.assertEqual(result['services'][name]['volumes'], ['/models:/models'])
        self.assertEqual(result['services']['api-3']['deploy']['labels']['caddy_0'], 'http://index-api')
        self.assertIn('caddy', self.before['services']['api-1']['deploy']['labels'])

    def test_runtime_drift_blocks_release(self):
        for key, value in [('volumes', []), ('environment', {'NVIDIA_VISIBLE_DEVICES': '0'})]:
            changed = copy.deepcopy(self.desired)
            changed['services']['api-3'][key] = value
            with self.assertRaises(AssertionError):
                module.build_transition(self.before, changed)

    def test_unexpected_service_cannot_be_pruned(self):
        self.before['services']['unrelated'] = {}
        with self.assertRaises(AssertionError):
            module.build_transition(self.before, self.desired)

    def test_upstream_verification_keeps_host_scope(self):
        config = {'apps': {'http': {'servers': {'server': {'routes': [
            {'match': [{'host': ['index-api']}], 'handle': [{'handler': 'subroute', 'routes': [
                {'handle': [{'handler': 'reverse_proxy', 'upstreams': [{'dial': 'index-tts_api-3.:8000'}]}]}]}]},
            {'match': [{'host': ['another']}], 'handle': [{'handler': 'reverse_proxy', 'upstreams': [{'dial': 'old:8000'}]}]},
        ]}}}}}
        self.assertEqual(module.index_upstreams(config), ['index-tts_api-3.:8000'])

    def test_environment_only_edit_blocks_update_without_any_put(self):
        with patch.object(module, 'request', return_value={'Name': 'index-tts', 'EndpointId': 4, 'Env': [{'name': 'setting', 'value': 'new'}]}) as request:
            with self.assertRaises(AssertionError):
                module.update('unchanged yaml', [{'name': 'setting', 'value': 'old'}], 'unused')
            self.assertEqual(request.call_count, 1)


if __name__ == '__main__':
    unittest.main()
