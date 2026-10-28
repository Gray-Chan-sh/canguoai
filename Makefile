PY ?= python3

.PHONY: plan env dry all report push stock clean

plan:
	$(PY) -m cgbench plan

env:
	$(PY) -m cgbench env

# validate the full 24h plan in minutes, without downloading any weights
dry:
	$(PY) -m cgbench all --dry-run

all:
	$(PY) -m cgbench all

report:
	$(PY) -m cgbench report

push:
	$(PY) tools/push.py --git --webhook

stock:
	$(PY) tools/canguo.py

day:
	bash run_day.sh

clean:
	rm -rf results
