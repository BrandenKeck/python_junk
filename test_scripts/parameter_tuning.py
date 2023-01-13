import json
import time
import pandas as pd
from test_model import test_model

class sk8r_lite():

    def __init__(self, filename):
        f = open(filename)
        loaded = json.load(f)
        self.id = loaded["id"]
        self.name = loaded["name"]
        self.data = pd.DataFrame.from_dict(json.loads(loaded["data"]))
        self.model = None
        self.set_columns()
        self.create_model()

    def set_columns(self):
        self.preX = [
            "g_toi", "a_toi", "s_toi", "h_toi", "ga_toi", "ta_toi",
            "gt3", "at3", "st3", "ht3", "gat3", "tat3",
            "gt5", "at5", "st5", "ht5", "gat5", "tat5",
            "gt10", "at10", "st10", "ht10", "gat10", "tat10",
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
            "oggat10", "ogsat10", "ogsvp10"
        ]
        self.postX = [["is_home", 2], ["tdr", 5], ["odr", 5]]

    def create_model(self):
        self.model = test_model(
            10,
            self.preX,
            self.postX,
            "goals",
            self.data
        )

# Test on guentzel
sk8r = sk8r_lite('data/Phil Kessel.json')
sk8r.model.set_numerical_data()
sk8r.model.set_categorical_data()
print(sk8r.model.X.shape)


# start = time.time()
# xx = sk8r.model.paramsearch()
# sk8r.model.train(xx)
# end = time.time()
# print(sk8r.model.params)
# print(f"Time: {end - start}")

