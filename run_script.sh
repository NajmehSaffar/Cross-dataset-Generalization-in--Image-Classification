#!/bin/bash

# Define the different arguments you want to pass to your Python script
# Modify the arguments according to your specific needs
args=(
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline1 --exp D1a"
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline1 --exp D1b"
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline1 --exp D2a"
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline1 --exp D2b" 
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline1 --exp D3a"
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline1 --exp D3b"
    
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline1 --exp D2D3 --cross_dataset"
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline1 --exp D1D3 --cross_dataset"
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline1 --exp D1D2 --cross_dataset"

    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline2 --exp D2D3 --cross_dataset"
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline2 --exp D1D3 --cross_dataset"
    #"--data_dir ../data/ --batch_size 256 --num_epochs 60 --im_size 32 --learning_rate 0.01 --cuda --setup Baseline2 --exp D1D2 --cross_dataset"
    
    "--data_dir ../data/ --batch_size 256 --num_epochs 200 --im_size 128 --learning_rate 0.0001 --cuda --setup SupCon --exp D2D3 --cross_dataset"
    "--data_dir ../data/ --batch_size 256 --num_epochs 200 --im_size 128 --learning_rate 0.0001 --cuda --setup SupCon --exp D1D3 --cross_dataset"
    "--data_dir ../data/ --batch_size 256 --num_epochs 200 --im_size 128 --learning_rate 0.0001 --cuda --setup SupCon --exp D1D2 --cross_dataset"
)

python_file_directory="scripts/"
# Change the working directory to the Python file directory
cd "$python_file_directory" || exit

# Create directory if it doesn't exist
out_dir="../outputs/"
mkdir -p "$out_dir"

# Loop through the arguments and execute your Python script for each argument
for arg in "${args[@]}"; do
    
    # Search for the value of "--setup" in the current argument and return the value
    setup_value=$(echo "$arg" | grep -oP "(?<=--setup\s)\w+")
    
    # Search for the value of "--exp" in the current argument and return the value
    exp_value=$(echo "$arg" | grep -oP "(?<=--exp\s)\w+")
    
    # Print the value of "--setup" for the current argument
    echo "---------- Setup value: $setup_value ,  Experiment value: $exp_value ---------- "
    
    # Save the output of the Python script to a file
    output_file="${out_dir}output-$setup_value-$exp_value.txt" # Create a unique output file name based on the argument
    #echo "$output_file"
    
    # Run your Python script with the current argument
    python train.py $arg >> $output_file
    
    # Print a separator for readability
    echo "---------------------------------------------------------------------"
done

