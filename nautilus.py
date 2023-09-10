import os
import sys
import re
import datetime
import itertools
import yaml
import tempfile
import termcolor
import string
import random
import codecs
import subprocess


def get_current_context():
	cmd = ["kubectl", "config", "current-context"]
	output = subprocess.check_output(cmd)
	return output.decode().strip()


def safe(name, length=8):
	name = re.sub(r'(\W)\1+', r'\1', re.sub(r'[^a-zA-Z0-9]', '-', name))
	if len(name) > length:
		name = name[:length]
	else:
		name += 'a' * (length - len(name))
	return name.lower()


def rot13(s):
	return codecs.encode(s, 'rot13')


def submit(args, name):
	uid = ''.join(random.choice(string.ascii_lowercase) for _ in range(3))
	context = get_current_context()
	cfg = dict(
		name='nh-'+rot13(f'{safe(name)}-{uid}').replace('-', ''),
		namespace=context,
		haosu='In' if 'haosu' in context else 'NotIn',
		user='nicklashansen',
		project='dreamerv3',
		image='latest',
		pvc='nh-fast-vol2',
		wandb_key='769a7a5e42ea92f54891f469925367f38ce6400f',
		cmd=f'python3 train.py {args}',
		cpu=2,
		mem=16,
		gpu=1,
	)
	with open('nautilus.yaml', 'r') as f:
		template = f.read()
	while '{{' in template:
		for key, value in cfg.items():
			regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
			template = re.sub(regexp, str(value), template)
	tmp = tempfile.NamedTemporaryFile(suffix='.yaml')
	with open(tmp.name, 'w') as f:
		f.write(template)
	print(termcolor.colored(f'{cfg["name"]}', 'yellow'), args if len(args) > 0 else 'None')
	os.system(f'kubectl create -f {tmp.name}')
	tmp.close()


def submit_batch(kwargs: dict):
	arg_list = list(itertools.product(*kwargs.values()))
	if len(arg_list) > 32:
		print(termcolor.colored(f'Error: {len(arg_list)} jobs exceeds limit of 32', 'red'))
		return
	print(termcolor.colored(f'Submitting {len(arg_list)} job{"s" if len(arg_list) > 1 else ""}', 'green'))
	for args in arg_list:
		args = ' '.join([f'{k}={v}' for k, v in zip(kwargs.keys(), args)])
		submit(args, name=kwargs['exp_name'][0] if 'exp_name' in kwargs else 'dev')


if __name__ == '__main__':
	kwargs = dict(arg.split('=') for arg in sys.argv[1:])
	kwargs = {k: v.split(',') for k, v in kwargs.items()}
	submit_batch(kwargs)
