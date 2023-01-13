import json
import numpy as np
import pandas as pd

###
# POPULATE CURRENT DATA
###

# Game Indicies
index = [ 2022020001, 2022020002, 2022020003 ]

# Known Rows - to be populated dynamically but not stored
g_toi = [1.02, 2.34, 5.04]
a_toi = [2.33, 2.56, 1.03]

# Unknown (TBD) Rows - to be populated later
opp_b_toi = np.zeros(len(g_toi)).tolist()

# Create Frame
data = np.array([ g_toi, a_toi, opp_b_toi ]).T.tolist()
columns = [ "g_toi", "a_toi", "opp_b_toi" ]
existing_frame = pd.DataFrame(
    index=index,
    data=data,
    columns=columns
)

###
# POPULATE PREVIOUS UNKNOWNS
###

existing_frame.loc[2022020001, 'opp_b_toi'] = 0.123
existing_frame.loc[2022020002, 'opp_b_toi'] = 0.456
existing_frame.loc[2022020003, 'opp_b_toi'] = 0.789

###
# APPEND NEW DATA
###

# Repeat data collection steps
new_index = [ 2022020004, 2022020005 ]
new_g_toi = [6.58, 0.25]
new_a_toi = [3.14, 0.55]
new_opp_b_toi = np.zeros(len(new_g_toi)).tolist()
new_data = np.array([ new_g_toi, new_a_toi, new_opp_b_toi ]).T.tolist()
columns = [ "g_toi", "a_toi", "opp_b_toi" ]
new_frame = pd.DataFrame(
    index=new_index,
    data=new_data,
    columns=columns
)
new_frame.loc[2022020004, 'opp_b_toi'] = 0.111
new_frame.loc[2022020005, 'opp_b_toi'] = 0.222

# Combine the frames
existing_frame = pd.concat([existing_frame, new_frame])

###
# JSON EXPORT
###

exported_frame = existing_frame.to_json()

###
# JSON IMPORT
###

imported_dict = json.loads(exported_frame)
imported_frame = pd.DataFrame.from_dict(imported_dict)

###
# INDEX CHECK
###

existing_frame
print(2022020001 in existing_frame.index)
print(2022020004 in existing_frame.index)
