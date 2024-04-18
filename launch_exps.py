"""Test exps launcher script

    Examples:
        python launch_exps.py script=script1 config=conf1 [<overwrite script-specific parameters>]  [host.<overwrite sbatch parameters>] [exps.fake=true]
        

"""
from exps_launcher.ExpsLauncher import ExpsLauncher

def main():
    expsLauncher = ExpsLauncher(root='exps_launcher_configs')
    expsLauncher.launch()

if __name__ == '__main__':
    main()