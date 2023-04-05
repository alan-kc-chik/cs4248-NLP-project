# BERT + BiLSTM + GAT 

## Infrastructure 
[NUS High Performance Computing (HPC)](https://nusit.nus.edu.sg/hpc/) 

## Setup
1. Copy the LUN data to `./data`
    * `./data/fulltrain.csv`
    * `./data/balancedtest.csv`
2. Run `./prepare_training_data.ipynb` to reduce the training data size
3. Connect to `atlas8.nus.edu.sg` over SFTP 
4. Upload the project code to `/hpctmp/yk/CS4248/GAT/`
5. Connect to the HPC cluster via SSH 
   ```commandline
   ssh e0741024@atlas8.nus.edu.sg
   ```
6. Execute the commands in `./EnvSetup.sh` to resolve package dependencies  

## Model Training and Testing 
1. Connect to the HPC cluster via SSH 
   ```
   ssh e0741024@atlas8.nus.edu.sg
   ```
2. Go the project directory 
   ```commandline
   cd /hpctmp/yk/CS4248/GAT/
   ```
3. Modify the job script `BatchJob.pbs` as needed to choose whether to run the training or testing 
4. Submit the job 
   ```commandline
   qsub BatchJob.pbs
   ``` 
5. Check the job status and make sure it completes successfully 
   ```commandline
   qstat -xfn
   ```
6. The logs produced during the training and testing are available at `./log`


## Note
The code is adapted from https://github.com/MysteryVaibhav/fake_news_semantics/. 

