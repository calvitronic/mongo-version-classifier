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
        pcap_fp = os.path.join(vers, 'tempytrnld.pcap')
        # Maybe we should make this togglable; pcapngs apparenlty contain more information than regular pcaps
        # so we'd technically be throwing out some data
        if os.path.isfile(pcap_fp):
            subprocess.run(["tshark", "-F", "pcap", "-r", pcap_fp, "-w", pcap_fp], stdout = subprocess.DEVNULL)
        n = num_packets(pcap_fp)
        condition_fp = os.path.join(vers, 'condition.json')
        if os.path.isfile(condition_fp):
            with open(os.path.join(vers, 'condition.json'), 'r') as jfile:
                json_dict = json.load(jfile)
                ver_from_json = json_dict['version_update']['mongo']
                new_ver = ""
                if ver_from_json == "latest":
                    new_ver = ver_from_json
                    if n in packets[4]:
                        packets[4][n] += 1
                    else: 
                        packets[4][n] = 1
                elif ver_from_json == "7.0":
                    new_ver = "seven"
                    if n in packets[4]:
                        packets[4][n] += 1
                    else: 
                        packets[4][n] = 1
                elif ver_from_json == "6.0":
                    new_ver = "six"
                    if n in packets[3]:
                        packets[3][n] += 1
                    else: 
                        packets[3][n] = 1
                elif ver_from_json == "5.0":
                    new_ver = "five"
                    if n in packets[2]:
                        packets[2][n] += 1
                    else: 
                        packets[2][n] = 1
                elif ver_from_json == "4.0":
                    new_ver = "four"
                    if n in packets[1]:
                        packets[1][n] += 1
                    else: 
                        packets[1][n] = 1
                elif ver_from_json == "3.0":
                    new_ver = "three"
                    if n in packets[0]:
                        packets[0][n] += 1
                    else: 
                        packets[0][n] = 1
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
    #packets = [[],[],[],[],[]]
    packets = [{}, {}, {}, {}, {}]

    # Step down into each of those repo directories, step into each of their subdirectories
    for repo in tqdm(repo_dir_list):
        handle_sub_directory(repo)

    ver = 3
    for c in packets:
        if len(c) != 0:
            # c = list(filter(lambda x: x > 0 and x < 100, c))
            plt.bar(list(c.keys()), list(c.values()), color='g')
            plt.title('Version ' + str(ver) + ' Packets')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show()

            total_count = sum(c.values())
            mean_value = sum(value * count for value, count in c.items()) / total_count

            sorted_values = sorted(c.items())
            cumulative_counts = 0
            median_value = None

            if total_count % 2 == 1:  # Odd number of elements
                middle = total_count // 2
                for value, count in sorted_values:
                    cumulative_counts += count
                    if cumulative_counts > middle:
                        median_value = value
                        break
            else:  # Even number of elements
                middle1 = total_count // 2 - 1
                middle2 = total_count // 2
                for value, count in sorted_values:
                    cumulative_counts += count
                    if cumulative_counts > middle1 and median_value is None:
                        median1 = value
                    if cumulative_counts > middle2:
                        median2 = value
                        median_value = (median1 + median2) / 2
                        break

            mode_value = max(c.items(), key=lambda item: item[1])[0]
            min_value = min(c.keys())
            max_value = max(c.keys())
            print(f'Version {ver} Stats:\n Repos: {total_count} Mean: {mean_value} Min: {min_value} Max: {max_value} Median: {median_value} Mode: {mode_value}')
            with open(f'version_{ver}.pkl', 'wb') as file:
                pickle.dump(c, file)

        ver += 1