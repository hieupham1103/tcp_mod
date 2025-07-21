import os
import re
import sys
import csv
import subprocess

TRAINER = "TCP_MOD_MMA"
results_file = f"results/tuning_{TRAINER}.csv"
EPOCH = [i for i in range(5, 105, 5)]
dataset = "dtd"
attempts_number = 3
print("EPOCH:", EPOCH)

def get_res(cmd):
    result = subprocess.check_output(cmd, shell=True).decode()
    base = re.search(r"\* base2new train_base accuracy: ([\d.]+)%", result)
    novel = re.search(r"\* base2new test_new accuracy: ([\d.]+)%", result)
    return base.group(1), novel.group(1)

def main():
    for epoch in EPOCH:
        for attempt in range(attempts_number):
            print(f"Running tuning for {TRAINER} on {dataset} with epoch {epoch}, attempt {attempt + 1}/{attempts_number}")
            subprocess.run(
                ["bash", "scripts/base2new/run.sh", dataset, str(epoch)],
                
            )
            base, novel = get_res(
                f"bash eval.sh {TRAINER}"
            )
            
            print(f"Base accuracy: {base}, Novel accuracy: {novel}")
            
            with open(results_file, "a", newline="") as f:
                if f.tell() == 0:
                    writer = csv.writer(f)
                    writer.writerow(["EPOCH", "BASE_ACC", "NOVEL_ACC"])
                writer = csv.writer(f)
                writer.writerow([epoch, base, novel])
if __name__ == "__main__":
    main()
    print("Tunning script executed successfully.")