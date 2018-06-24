#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json("results/log")
df.plot(x="iteration", y=["main/loss", "main/loss/conf", "main/loss/loc"], ylim=(0, 12))
plt.savefig("loss.pdf")
plt.show()
