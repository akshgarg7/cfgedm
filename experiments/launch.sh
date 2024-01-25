#!/bin/bash

MODEL="egnn_dynamics"
DROP_PROB="0.1"
DATASET_PORTION="0.5"
EPOCHS="3000"
BATCH_SIZE="32"
STABILITY_SAMPLES="500"
GUIDANCE_WEIGHT="0.25" # Replace 'some_value' with the actual value

properties=('alpha' 'homo' 'lumo' 'gap' 'mu' 'Cv')

for property in "${properties[@]}"
do
    job_name="job_${property}"
    sed "s/JOB_NAME_PLACEHOLDER/${job_name}/g; \
         s/MODEL_PLACEHOLDER/${MODEL}/g; \
         s/EPOCHS_PLACEHOLDER/${EPOCHS}/g; \
         s/STABILITY_SAMPLES_PLACEHOLDER/${STABILITY_SAMPLES}/g; \
         s/BATCH_SIZE_PLACEHOLDER/${BATCH_SIZE}/g; \
         s/GUIDANCE_WEIGHT_PLACEHOLDER/${GUIDANCE_WEIGHT}/g; \
         s/DROP_PROB_PLACEHOLDER/${DROP_PROB}/g; \
         s/TEST_EPOCHS_PLACEHOLDER/${TEST_EPOCHS}/g; \
         s/DATASET_PORTION_PLACEHOLDER/${DATASET_PORTION}/g; \
         s/PROPERTY_PLACEHOLDER/${property}/g" experiments/job_template.sh > temp_job_${property}.sh

    sbatch temp_job_${property}.sh
    # rm temp_job_${property}.sh  # Optional: Remove the temporary job file after submission
done
