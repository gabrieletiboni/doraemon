from collections import Mapping
import os
import pdb
import random
import string

try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError(f"Package omegaconf not installed.")

class OmegaConfParser():
    """Handler for parsing CLI parameters through OmegaConf"""
    def __init__(self):
        self.params_as_list = ['config'] 

    def parse_from_cli(self):
        args = OmegaConf.from_cli()  # Read the cli args
        args = self.pars_as_list(args, self.params_as_list)
        return args
    
    def add_extension(self, config_file):
        assert type(config_file) == str
        filename, _ = os.path.splitext(config_file)  # Returns filename and extension
        return filename+".yaml"

    def pformat_dict(self, d, indent=0):
        """Print OmegaConf args
            print(pformat_dict(args, indent=0))
        """
        fstr = ""
        for key, value in d.items():
            fstr += '\n' + '  ' * indent + str(key) + ":"
            if isinstance(value, Mapping):
                fstr += self.pformat_dict(value, indent+1)
            else:
                fstr += ' ' + str(value)
        return fstr

    def to_dict(self, args):
        return OmegaConf.to_container(args)

    def as_list(self, arg):
        if isinstance(arg, str) or isinstance(arg, int) or isinstance(arg, float):
            return [arg]
        elif isinstance(arg._content, list):
            return arg
        else:
            raise ValueError(f"This parameter was neither a string nor a list: {arg}")

    def pars_as_list(self, args, keys):
        for key in keys:
            if key in args:
                args[key] = self.as_list(args[key])
        return args

    def save_config(self, config, path, filename='config.yaml'):
        with open(os.path.join(path, filename), 'w', encoding='utf-8') as file:
            OmegaConf.save(config=config, f=file.name)
        return

    def create_dirs(self, path):
        try:
            os.makedirs(os.path.join(path))
        except OSError as error:
            pass

    def get_random_string(self, n=5):
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

    @staticmethod
    def is_list(val):
        if isinstance(val, list):
            return True
        elif hasattr(val, '_content') and isinstance(val._content, list):
            return True
        else:
            return False

    @staticmethod
    def is_boolean(val):
        if isinstance(val, bool):
            return True
        elif hasattr(val, '_content') and isinstance(val._content, bool):
            return True
        else:
            return False