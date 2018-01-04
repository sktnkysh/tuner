import subprocess
from tuner import net


def exec_hyperas(train_dir, validation_dir, model):
    template_fname = 'template_base_hyperas.py'
    with open(template_fname, 'r') as f:
        template_code = f.read()

    code_hyperas = template_code.format(
        train_dir=train_dir, validation_dir=validation_dir, model=model.__name__)

    fname = 'code_hyperas.py'
    with open(fname, 'w') as f:
        f.write(code_hyperas)

    subprocess.run(['python', fname])


if __name__ == '__main__':
    train_dir = '../examples/dataset/brain/train'
    validation_dir = '../examples/dataset/brain/validation'
    exec_hyperas(train_dir, validation_dir, net.aug)
