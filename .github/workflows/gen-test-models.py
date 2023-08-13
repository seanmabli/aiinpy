tests = ["cnn-mnist",
"lstm-timeseries",
"gru-timeseries",
"rnn-timeseries",
"nn-andor",
"nn-nonlinearity",
"rnn-posnegcon",
]

start = """
"""

for i in range(len(tests)):
    template = """
name: test {{test}}
on: [push]
jobs:
  {{test}}:
    runs-on: ubuntu-latest
    steps:
      - name: clone
        uses: actions/checkout@v3
      
      - name: setup python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10' # install the python version needed

      - name: install python packages
        run: |
          python3 -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: run tests
        run: cd example && python3 {{test}}.py

    """
    template = template.replace("{{test}}", tests[i])
    start += template

with open("test-models.yml", "w") as f:
    f.write(start)