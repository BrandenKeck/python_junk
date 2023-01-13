# Standard Imports (some for testing)
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

# Import Models
from src.lakshmi import lakshmi
ll = lakshmi()
ll.load("data.json")

columns = [ "date", "team", "opp_team", "toi", "goals", "assists", "shots",
                "g_toi", "a_toi", "s_toi", "b_toi", "h_toi", "ga_toi", "ta_toi", "is_home",
                 "toi3", "gt3", "at3", "st3", "bt3", "ht3", "gat3", "tat3",
                 "toi5", "gt5", "at5", "st5", "bt5", "ht5", "gat5", "tat5",
                 "toi10", "gt10", "at10", "st10", "bt10", "ht10", "gat10", "tat10",
                 "toi82", "gt82", "at82", "st82", "bt82", "ht82", "gat82", "tat82",
                 "tdr", "odr",
                 "tsgt1", "tsat1", "tsst1", "tstat1", "tsgat1",
                 "osgat1", "ostat1", "osbt1", "osht1", "odbt1", "odht1",
                 "oggat1", "ogsat1", "ogsvp1",
                 "tsgt3", "tsat3", "tsst3", "tstat3", "tsgat3",
                 "osgat3", "ostat3", "osbt3", "osht3", "odbt3", "odht3",
                 "oggat3", "ogsat3", "ogsvp3",
                 "tsgt5", "tsat5", "tsst5", "tstat5", "tsgat5",
                 "osgat5", "ostat5", "osbt5", "osht5", "odbt5", "odht5",
                 "oggat5", "ogsat5", "ogsvp5",
                 "tsgt10", "tsat10", "tsst10", "tstat10", "tsgat10",
                 "osgat10", "ostat10", "osbt10", "osht10", "odbt10", "odht10",
                 "oggat10", "ogsat10", "ogsvp10",
                 "tsgt82", "tsat82", "tsst82", "tstat82", "tsgat82",
                 "osgat82", "ostat82", "osbt82", "osht82", "odbt82", "odht82",
                 "oggat82", "ogsat82", "ogsvp82" ]
dat = pd.DataFrame(columns=columns)

for idx in ll.skaters:
    sk = ll.skaters[idx]
    avg_goals = np.mean(sk.data.loc[:, "goals"])
    avg_assists = np.mean(sk.data.loc[:, "assists"])
    avg_shots = np.mean(sk.data.loc[:, "shots"])
    if avg_goals < 0.1 and avg_assists < 0.1 and avg_shots < 0.5:
        dat = pd.concat([dat, sk.data])

from src.zip_network import zip_model
from src.skater import skater
ss = skater(00000000, "Dummy")
ss.data = dat
ss.data = ss.data.reset_index()
ss.generate_models()
ss.predict_last()

ll.skaters[00000000] = ss
ll.export("data.json")
