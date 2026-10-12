import glob
import os
import unittest

from ladder.components.inference.api.client.run_model import RunnerConfig
from ladder.core.databib.config import ConfigParser


class ConfigParseTest(unittest.TestCase):
    configs_dir = 'ladder/components/inference/models/vlams/configs'
    exclude_dirs = ['.venv']

    def test_config_parses(self):
        found_config_files = False

        for config_path in glob.iglob(os.path.join(self.configs_dir, '**/*.yaml'), recursive=True):
            if any(exclude in config_path for exclude in self.exclude_dirs):
                continue
            found_config_files = True
            print(f'Parsing: {config_path}')
            ConfigParser(RunnerConfig).parse_args(args=["--config", config_path])

        if not found_config_files:
            raise FileNotFoundError('Did not find any config files to test.')


if __name__ == '__main__':
    unittest.main()
