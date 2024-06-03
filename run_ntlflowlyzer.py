import os
import json
from scapy.all import rdpcap
import matplotlib.pyplot as plt
from statistics import mode
import numpy as np
from tqdm import tqdm
import pickle
import subprocess

def handle_sub_directory(repo_name):
    # Get a list of this one's subdirectories
    repo_rel_path = os.path.join('.', repo_name)
    if not os.path.isdir(repo_rel_path):
        return
    old_version_names = os.listdir(path=repo_rel_path)
    
    # Sloppy, but it does the job
    for ind in range(len(old_version_names)):
        old_version_names[ind] = os.path.join(repo_rel_path, old_version_names[ind])

    # Iterate through those subdirectories, load the conditions.json, and retrieve the version
    for vers in old_version_names:
        pcap_fp = os.path.join(vers, 'tempytrnld.pcap')
        if (not os.path.isfile(pcap_fp)) or (os.path.getsize(pcap_fp) < 25):
            continue
        condition_fp = os.path.join(vers, 'condition.json')
        if os.path.isfile(condition_fp):
            with open(os.path.join(vers, 'condition.json'), 'r') as jfile:
                json_dict = json.load(jfile)
                ver_from_json = json_dict['version_update']['mongo']
                new_ver = ""
                if ver_from_json == "latest":
                    new_ver = ver_from_json
                    
                elif ver_from_json == "7.0":
                    new_ver = "seven"
                    
                elif ver_from_json == "6.0":
                    new_ver = "six"
                    
                elif ver_from_json == "5.0":
                    new_ver = "five"
                    
                elif ver_from_json == "4.0":
                    new_ver = "four"
                    
                elif ver_from_json == "3.0":
                    new_ver = "three"
                    
                else:
                    print("Unable to determine version.")
                # Now that we have the version, let's build the json
                config_to_json = {"pcap_file_address": pcap_fp, "output_file_address": os.path.join("..", os.path.join("csvs", repo_name + new_ver + ".csv"), "label": new_ver)}
                with open('tmp_file.json', 'w') as jfile:
                    json.dump(config_to_json, jfile)
                prog = subprocess.run(["ntlflowlyzer", "-c", "tmp_file.json"], stdout=subprocess.DEVNULL)
                
    return

def num_packets(pcap_file):
    try:
        # Read packets from the pcap file
        packets = rdpcap(pcap_file)
        # Return packets
        return len(packets)

    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    # First get the list of all the repo directories
    # Use this python script INSIDE of the directory with all the repo subdirectories
    repo_dir_list = os.listdir(path='.')
    csv_directory = os.path.join("..", "csvs")

    # Step down into each of those repo directories, step into each of their subdirectories
    for repo in tqdm(repo_dir_list):
        handle_sub_directory(repo)
