import subprocess

date = '02_11'
# conda_activate = 'source /scratch/slerman/miniconda/bin/activate agi'
conda_activate = 'module load anaconda3/5.2.0b'

for num_ways in [7, 230]:
    for model in ['cnn', 'cnnp', 'dnn', 'logreg']:
        script = f"""#!/bin/bash
#SBATCH -c 11
#SBATCH -t 5-00:00:00 -o ./{date}_{num_ways}.log -J {date}_{num_ways}
#SBATCH --mem=80gb 

{conda_activate}
python3 train.py --log-dir {date}_{num_ways} --num-classes {num_ways} --num-workers 10 --name {model}"""

        # Write script
        with open("sbatch_script", "w") as file:
            file.write(script)

        # Launch script (with error checking / re-launching)
        success = "error"
        while "error" in success:
            try:
                success = str(subprocess.check_output(['sbatch {}'.format("sbatch_script")], shell=True))
                print(success[2:][:-3])
                if "error" in success:
                    print("Errored... trying again")
            except:
                success = "error"
                if "error" in success:
                    print("Errored... trying again")
        print("Success!")
