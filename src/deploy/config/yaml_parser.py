from ruamel.yaml import YAML


class YamlParser:
    def __init__(self, path):
        yaml = YAML()
        self._config = {}
        with open(path, "r") as stream:
            for data in yaml.load_all(stream):
                if isinstance(data, dict):
                    self._config.update(data)

    def get_config(self):
        return self._config
