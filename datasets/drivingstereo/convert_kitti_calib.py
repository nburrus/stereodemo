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

# parse numpy array from a string a b c d e f g h i
def parse_numpy_array(s):
    return np.fromstring(s, sep=" ")

fields = {}

input_path = Path(sys.argv[1])

with open(input_path) as f:    
    for l in f:
        kv = l.split(':')
        k, v = kv
        fields[k] = v.strip()

print (fields)

w0, h0 = parse_numpy_array(fields['S_rect_101']).reshape(2)
w1, h1 = parse_numpy_array(fields['S_rect_103']).reshape(2)
assert w0 == w1 and h0 == h1

K0 = parse_numpy_array(fields['P_rect_101']).reshape(3, 4)
print (K0)

K1 = parse_numpy_array(fields['P_rect_103']).reshape(3, 4)
print (K1)

T = parse_numpy_array(fields['T_103']).reshape(3)

assert (K0[0,0] == K1[0,0])

# https://stackoverflow.com/a/61684187/1737680
# P(i)rect = [[fu 0  cx  -fu*bx],
#            [0  fv  cy -fv*by],
#            [0   0   1  0]]
# baseline = -K1[0,3]/K1[0,0] # does not work, ~2x too large
baseline = np.linalg.norm(T)
calib = Calibration(int(w0), int(h0), K0[0,0], K0[1,1], K0[0,2], K1[0,2], K0[1,2], baseline_meters=baseline)
print (calib)

output_json = input_path.parent / 'stereodemo_calibration.json'
with open(output_json, 'w') as f:
    f.write (calib.to_json())
