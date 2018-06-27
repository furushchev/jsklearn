#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json("results/log")
df.plot(x="iteration", y=["main/loss"])
plt.savefig("loss.pdf")
plt.show()
