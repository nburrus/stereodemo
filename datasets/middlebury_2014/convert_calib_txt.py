#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path
import sys
import json
import re

import numpy as np

@dataclass
class Calibration:
    width: int
    height: int
    fx: float
    fy: float
    cx0: float
    cx1: float
    cy: float
    baseline_meters: float

    def to_json(self):
        return json.dumps(self.__dict__)

    def from_json(json_str):
        d = json.loads(json_str)
        return Calibration(**d)

# parse numpy array from a string [a b c ; d e f ; g h i]
def parse_numpy_array(s):
    s = s.replace("[", "").replace("]", "").replace(";", " ")
    return np.fromstring(s, sep=" ").reshape(3,3)

fields = {}

input_path = Path(sys.argv[1])

with open(input_path) as f:    
    for l in f:
        kv = l.split('=')
        k, v = kv
        fields[k] = v.strip()

print (fields)

K0 = parse_numpy_array(fields['cam0'])
print (K0)

K1 = parse_numpy_array(fields['cam1'])
print (K1)

assert np.count_nonzero(K0 != K1) <= 1 # only cx can differ

calib = Calibration(int(fields['width']), int(fields['height']), K0[0,0], K0[1,1], K0[0,2], K1[0,2], K0[1,2], float(fields['baseline'])*1e-3)
print (calib)

output_json = input_path.parent / 'stereodemo_calibration.json'
with open(output_json, 'w') as f:
    f.write (calib.to_json())
