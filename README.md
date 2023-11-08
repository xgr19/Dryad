# Dryad: Deploying Adaptive Trees on Programmable Switches for Networking Classification (ICNP2023)

**More information about us** [https://xgr19.github.io](https://xgr19.github.io)  

## Code Architecture

```
-- hardware_configure
	-- create_p4_file.py (create ODT.p4 for a specific setting)
  -- template (folder for P4 code templates)
		
-- model_data
  -- first_soft_pruned_tree.json (the original trained ODT)
  -- x_test.pkl and y_test.pkl (input samples and their labels for testing a pruned ODT)

-- install_process.py (the progressive search and compiler for the OpenMesh switch)
-- prune_util.py (helpful functions to conduct pruning, and loading model/test data, etc.)

```

## Train the ODT (lin-jy22@mails.tsinghua.edu.cn)  

wait for adding ...

## Run progressive search & generate P4
1. run install_process.py, the code will output the settings for a suitable ODT (table arrangement type, bit, depth) and the P4 table entries for the ODT
2. run hardware_configure/create_p4_file.py with the settings (table arrangement type, bit, depth), the code outputs the desired ODT.p4


