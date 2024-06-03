import os
import json
from scapy.all import rdpcap
import matplotlib.pyplot as plt
from statistics import mode
import numpy as np
from tqdm import tqdm

def handle_sub_directory(repo_name):
    global packets
    # Get a list of this one's subdirectories
    repo_rel_path = os.path.join('.', repo_name)
    # print(repo_rel_path)
    if not os.path.isdir(repo_rel_path):
        return
    old_version_names = os.listdir(path=repo_rel_path)
    if ".DS_Store" in old_version_names:
        old_version_names.remove(".DS_Store")

    # Sloppy, but it does the job
    for ind in range(len(old_version_names)):
        old_version_names[ind] = os.path.join(repo_rel_path, old_version_names[ind])

    # Iterate through those subdirectories, load the conditions.json, and retrieve the version
    for vers in old_version_names:
        n = num_packets(os.path.join(vers, 'tempytrnld.pcap'))
        condition_fp = os.path.join(vers, 'condition.json')
        if os.path.isfile(condition_fp):
            with open(os.path.join(vers, 'condition.json'), 'r') as jfile:
                json_dict = json.load(jfile)
                ver_from_json = json_dict['version_update']['mongo']
                new_ver = ""
                if ver_from_json == "latest":
                    new_ver = ver_from_json
                    packets[4].append(n)
                elif ver_from_json == "7.0":
                    new_ver = "seven"
                    packets[4].append(n)
                elif ver_from_json == "6.0":
                    new_ver = "six"
                    packets[3].append(n)
                elif ver_from_json == "5.0":
                    new_ver = "five"
                    packets[2].append(n)
                elif ver_from_json == "4.0":
                    new_ver = "four"
                    packets[1].append(n)
                elif ver_from_json == "3.0":
                    new_ver = "three"
                    packets[0].append(n)
                else:
                    print("Unable to determine version.")
                if not new_ver == "":
                    os.rename(vers, os.path.join(repo_rel_path, new_ver))
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
    packets = [[],[],[],[],[]]

    # Step down into each of those repo directories, step into each of their subdirectories
    for repo in tqdm(repo_dir_list):
        handle_sub_directory(repo)

    ver = 3
    for c in packets:
        if len(c) != 0:
            c = list(filter(lambda x: x > 0, c))
            plt.hist(c, bins=10, edgecolor='black')
            plt.title('Version ' + str(ver) + ' Packets')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show()
            print(f"Version {ver} Stats:\n Repos: {len(c)} Mean: {sum(c)/float(len(c))} Min: {min(c)} Max: {max(c)} Median: {c[int(len(c)/2)]} Mode: {mode(c)}")
        ver += 1