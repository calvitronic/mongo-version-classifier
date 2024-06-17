# MOVERC: MongoDB Version Classifier

MOVERC is a random forest classifier that takes features extracted from NTLFlowLyzer as input and outputs classification reports for the test and validation datasets, as well as Trustee's model explanation global fidelity report and model explanation score report. 

Our first step was to use the DYNAMO paper's list of orchestable repositories from https://zenodo.org/records/7194189 by downloading the `list_of_executed_repositories.csv`. Please run the file `mongo_repos.ipynb` to extract the list of MongoDB repositories (those that have some variation of the word 'mongoDB' in the repository's name) and drop the repositories that produced 0 packets. 

We then ran UCSB's netMosaic platform with these repositories - first using 2 MongoDB versions (versions 4.0 and 7.0) and then using 5 MongoDB versions (versions 3.0, 4.0, 5.0, 6.0, and 7.0). When using 2 versions, we made sure the repositories were building using `docker compose up` (~100 repositories). For 5 versions, we did not ensure the repositories were building, but established that repositories that produced packets by running netMosaic did not build (~300 repositories). We also ran these ~300 repositories with 3 different network traffic conditions: `*-*-27017-*-6: loss 65% delay 5.14ms rate 194409.26kbps`, `*-*-27017-*-6: loss 8% delay 3.25ms rate 988892.96kbps`, and `*-*-27017-*-6: loss 22% delay 8.11ms rate 195496.09kbps`. `*-*-27017-*-6` applies network conditions to anything that is destination port 27017 (MongoDB uses this port), `6` means TCP, and delay/rate are the specific network conditions on `*-*-27017-*-6`. 

Please run `rename_subdirs.py` on the results from netMosaic to rename the directories. Run `run_ntlflowlyzer.py` to get features from the netMosaic results. 

After, please run `classifier_Neural_Network.py` on the features for the neural network model, and `classifier_RF.py` for the random forest model (MOVERC), using the three different datasets. The first dataset uses ~100 repositories with binary version classification, the second dataset uses ~300 repositories with multi-version classification (5 versions), and the last dataset combines the first and second datasets. Please run `combine_repo_csvs.py` to combine the datasets together. 

The resulting decision trees, *dt_explanation.pdf* and *pruned_dt_explanation.pdf*, from Trustee's output will automatically be saved in the same directory. 
