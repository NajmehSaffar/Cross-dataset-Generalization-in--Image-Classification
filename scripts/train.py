import time
import argparse
import pandas as pd
from dataset import load_data
from Baseline1.trainer import Trainer as Trainer1
from Baseline2.trainer import Trainer as Trainer2
from SupCon.trainer import Trainer as Trainer3
from utils import *

def parse_arguments_from_terminal():
    parser = argparse.ArgumentParser(description='PyTorch Model Training')
    parser.add_argument('--data_dir', type=str, default='../data/', help='Path to the data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--im_size', type=int, default=32, help='Size of images')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for training')
    parser.add_argument('--seed', type=int, default=30, help='Random seed for reproducibility')
    parser.add_argument('--display', action='store_true', help='Display the plots')
    parser.add_argument('--cross_dataset', action='store_true', help='Cross dataset or Within dataset?')
    parser.add_argument('--setup', type=str, default='Baseline1', help='Setup')
    parser.add_argument('--exp', type=str, default='D1a', help='Experiment')
    return parser.parse_args()


def main(args):
    
    # Load the dataframe from a file and return the loaded dataframe
    [df1_trn, df1_vld, df1_a, df1_b], [df2_trn, df2_vld, df2_a, df2_b], [df3_trn, df3_vld, df3_a, df3_b] = load_data(args)
    
    # ------------------------------------------------------------------------------
    # -------------------- Within-dataset evaluation --------------------
    # ------------------------------------------------------------------------------
    if not args.cross_dataset: 
        if (args.setup == "Baseline1") & (args.exp == "D1a"):
            trainer = Trainer1(df1_a, df1_b, None, None, args) # Instantiate the trainer
        elif (args.setup == "Baseline1") & (args.exp == "D1b"):
            trainer = Trainer1(df1_b, df1_a, None, None, args) # Instantiate the trainer
        elif (args.setup == "Baseline1") & (args.exp == "D2a"):
            trainer = Trainer1(df2_a, df2_b, None, None, args) # Instantiate the trainer
        elif (args.setup == "Baseline1") & (args.exp == "D2b"):
            trainer = Trainer1(df2_b, df2_a, None, None, args) # Instantiate the trainer   
        elif (args.setup == "Baseline1") & (args.exp == "D3a"):
            trainer = Trainer1(df3_a, df3_b, None, None, args) # Instantiate the trainer 
        elif (args.setup == "Baseline1") & (args.exp == "D3b"):
            trainer = Trainer1(df3_b, df3_a, None, None, args) # Instantiate the trainer 
        else:
            raise ValueError("Invalid experiment value.")
        
        print(f'==================== Start training {args.exp} ==================== ')
        trainer.train(trainer.ldr_trn, args)
        
        name_of_set = args.exp
        print(f'==================== Start validating for training set {name_of_set} ==================== ')
        trainer.validate(trainer.ldr_trn, args, name_of_set + '-train')
        
        name_of_set = name_of_set.replace('b', 'x').replace('a', 'b').replace('x', 'a') 
        print(f'==================== Start validating for testing set {name_of_set} ==================== ')
        trainer.validate(trainer.ldr_vld, args, name_of_set + '-test')
              
     
    else:
        # ------------------------------------------------------------------------------
        # -------------------- Cross-dataset evaluation (Baseline1) --------------------
        # ------------------------------------------------------------------------------
        if args.setup == "Baseline1":
            if args.exp == "D2D3":
                df_trn, df_vld = pd.concat([df2_trn, df3_trn]), pd.concat([df2_vld, df3_vld])
                trainer = Trainer1(df_trn, df_vld, df1_b, df1_a, args) # Instantiate the trainer
            elif args.exp == "D1D3":
                df_trn, df_vld = pd.concat([df1_trn, df3_trn]), pd.concat([df1_vld, df3_vld])
                trainer = Trainer1(df_trn, df_vld, df2_b, df2_a, args) # Instantiate the trainer
            elif args.exp == "D1D2":
                df_trn, df_vld = pd.concat([df1_trn, df2_trn]), pd.concat([df1_vld, df2_vld])
                trainer = Trainer1(df_trn, df_vld, df3_b, df3_a, args) # Instantiate the trainer
            else:
                raise ValueError("Invalid experiment value.")    
                
            print(f'==================== Start training {args.exp} ==================== ')
            trainer.train(trainer.ldr_trn, args)
            
            name_of_set = args.exp
            print(f'==================== Start validating for training set {name_of_set} ==================== ')
            trainer.validate(trainer.ldr_trn, args, name_of_set + '-train')
            print(f'==================== Start validating for validation set {name_of_set} ==================== ')
            trainer.validate(trainer.ldr_vld, args, name_of_set + '-valid')
            
            replacements = {"D2D3": "D1", "D1D3": "D2", "D1D2": "D3"}
            for pattern, replacement in replacements.items(): 
                name_of_set = name_of_set.replace(pattern, replacement)
            print(f'==================== Start validating for testing set {name_of_set}b ==================== ')
            trainer.validate(trainer.ldr_tstb, args, name_of_set + 'b-test')
            print(f'==================== Start validating for testing set {name_of_set}a ==================== ')
            trainer.validate(trainer.ldr_tsta, args, name_of_set + 'a-test')
           
        # ------------------------------------------------------------------------------
        # -------------------- Cross-dataset evaluation (Baseline2) --------------------
        # ------------------------------------------------------------------------------
        elif args.setup == "Baseline2":
            if args.exp == "D2D3":
                df2_trn["dataset_membership"] = 0
                df3_trn["dataset_membership"] = 1
                df_trn, df_vld = pd.concat([df2_trn, df3_trn]), pd.concat([df2_vld, df3_vld])
                trainer = Trainer2(df_trn, df_vld, df1_b, df1_a, args) # Instantiate the trainer
            elif args.exp == "D1D3":
                df1_trn["dataset_membership"] = 0
                df3_trn["dataset_membership"] = 1
                df_trn, df_vld = pd.concat([df1_trn, df3_trn]), pd.concat([df1_vld, df3_vld])
                trainer = Trainer2(df_trn, df_vld, df2_b, df2_a, args) # Instantiate the trainer
            elif args.exp == "D1D2":
                df1_trn["dataset_membership"] = 0
                df2_trn["dataset_membership"] = 1
                df_trn, df_vld = pd.concat([df1_trn, df2_trn]), pd.concat([df1_vld, df2_vld])
                trainer = Trainer2(df_trn, df_vld, df3_b, df3_a, args) # Instantiate the trainer 
            else:
                raise ValueError("Invalid experiment value.")   
                
            print(f'==================== Start training {args.exp} ==================== ')
            trainer.train(trainer.ldr_trn, args)
            
            name_of_set = args.exp
            print(f'==================== Start validating for training set {name_of_set} ==================== ')
            trainer.validate(trainer.ldr_trn, args, name_of_set + '-train')
            print(f'==================== Start validating for validation set {name_of_set} ==================== ')
            trainer.validate(trainer.ldr_vld, args, name_of_set + '-valid')
            
            replacements = {"D2D3": "D1", "D1D3": "D2", "D1D2": "D3"}
            for pattern, replacement in replacements.items(): 
                name_of_set = name_of_set.replace(pattern, replacement)
            print(f'==================== Start validating for testing set {name_of_set}b ==================== ')
            trainer.validate(trainer.ldr_tstb, args, name_of_set + 'b-test')
            print(f'==================== Start validating for testing set {name_of_set}a ==================== ')
            trainer.validate(trainer.ldr_tsta, args, name_of_set + 'a-test')

        # ------------------------------------------------------------------------------
        # -------------------- Cross-dataset evaluation (SupCon) --------------------
        # ------------------------------------------------------------------------------
        elif args.setup == "SupCon":
        
            if args.exp == "D2D3":
                df_trn, df_vld = pd.concat([df2_trn, df3_trn]), pd.concat([df2_vld, df3_vld])
                trainer = Trainer3(df_trn, df_vld, df1_b, df1_a, args) # Instantiate the trainer
            elif args.exp == "D1D3":
                df_trn, df_vld = pd.concat([df1_trn, df3_trn]), pd.concat([df1_vld, df3_vld])
                trainer = Trainer3(df_trn, df_vld, df2_b, df2_a, args) # Instantiate the trainer
            elif args.exp == "D1D2":
                df_trn, df_vld = pd.concat([df1_trn, df2_trn]), pd.concat([df1_vld, df2_vld])
                trainer = Trainer3(df_trn, df_vld, df3_b, df3_a, args) # Instantiate the trainer
            else:
                raise ValueError("Invalid experiment value.")
        
            print(f'==================== Start training {args.exp} ==================== ')
            trainer.train([trainer.ldr_trn_neg, trainer.ldr_trn_pos], args)
            
            name_of_set = args.exp
            print(f'==================== Start validating for training set {name_of_set} on Enc Output ==================== ')
            trainer.validate(trainer.ldr_trn, trainer.normal_vec_enc, args, name_of_set + '-train')
            print(f'==================== Start validating for training set {name_of_set} on Proj Output ==================== ')
            trainer.validate(trainer.ldr_trn, trainer.normal_vec_proj, args, name_of_set + '-train', True)
            print(f'==================== Start validating for validation set {name_of_set} on Enc Output ==================== ')
            trainer.validate(trainer.ldr_vld, trainer.normal_vec_enc, args, name_of_set + '-valid')
            print(f'==================== Start validating for validation set {name_of_set} on Proj Output ==================== ')
            trainer.validate(trainer.ldr_vld, trainer.normal_vec_proj, args, name_of_set + '-valid', True)
            
            replacements = {"D2D3": "D1", "D1D3": "D2", "D1D2": "D3"}
            for pattern, replacement in replacements.items(): 
                name_of_set = name_of_set.replace(pattern, replacement)
            print(f'==================== Start validating for testing set {name_of_set}b on Proj Output ==================== ')
            trainer.validate(trainer.ldr_tstb, trainer.normal_vec_proj, args, name_of_set + 'b-test-proj', True)
            print(f'==================== Start validating for testing set {name_of_set}a on Proj Output ==================== ')
            trainer.validate(trainer.ldr_tsta, trainer.normal_vec_proj, args, name_of_set + 'a-test-proj', True)  
            print(f'==================== Start validating for testing set {name_of_set}b on Enc Output ==================== ')
            trainer.validate(trainer.ldr_tstb, trainer.normal_vec_enc, args, name_of_set + 'b-test-enc')
            print(f'==================== Start validating for testing set {name_of_set}a on Enc Output ==================== ')
            trainer.validate(trainer.ldr_tsta, trainer.normal_vec_enc, args, name_of_set + 'a-test-enc')  
            
        
        else:
            raise ValueError("Invalid setup value.")   
            
        
if __name__ == '__main__':
    args = parse_arguments_from_terminal()
    
    start = time.time()
    main(args)
    end = time.time()
    days, hours, minutes, seconds = getTime(end-start)
    print(f"\n{int(days)} day(s) {int(hours)} hour(s) {int(minutes)} minute(s) {int(seconds)} second(s)")
    
