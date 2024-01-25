#!/bin/bash

MODEL="egnn_dynamics"
DROP_PROB="0.1"
DATASET_PORTION="0.5"
EPOCHS="3000"
BATCH_SIZE="32"
STABILITY_SAMPLES="500"
GUIDANCE_WEIGHT="0.25" # Replace 'some_value' with the actual value

# Define TEST_EPOCHS if required
# TEST_EPOCHS="some_value"

properties=('alpha,mu' 'mu,gap' 'mu,Cv')

for property in "${properties[@]}"; do
    IFS=',' read -r property1 property2 <<< "$property"

    echo property1: $property1
    echo property2: $property2  

    job_name="job_${property1}_${property2}_log"
    sed "s/JOB_NAME_PLACEHOLDER/${property1}_${property2}/g; \
         s/MODEL_PLACEHOLDER/${MODEL}/g; \
         s/EPOCHS_PLACEHOLDER/${EPOCHS}/g; \
         s/STABILITY_SAMPLES_PLACEHOLDER/${STABILITY_SAMPLES}/g; \
         s/BATCH_SIZE_PLACEHOLDER/${BATCH_SIZE}/g; \
         s/GUIDANCE_WEIGHT_PLACEHOLDER/${GUIDANCE_WEIGHT}/g; \
         s/DROP_PROB_PLACEHOLDER/${DROP_PROB}/g; \
         s/TEST_EPOCHS_PLACEHOLDER/${TEST_EPOCHS}/g; \
         s/DATASET_PORTION_PLACEHOLDER/${DATASET_PORTION}/g; \
         s/PROPERTY1_PLACEHOLDER/${property1}/g; \
         s/PROPERTY2_PLACEHOLDER/${property2}/g" experiments/two_template.sh > temp_job_${property1}_${property2}.sh

    bash temp_job_${property1}_${property2}.sh
    # rm temp_job_${property1}_${property2}.sh  # Optional: Remove the temporary job file after submission
done
