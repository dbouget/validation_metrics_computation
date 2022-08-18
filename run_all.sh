#! /bin/bash

#python main.py -c main_config_exp1.ini 2>&1 | tee ./logs/validation/exp1.txt
#python main.py -c main_config_exp2.ini 2>&1 | tee ./logs/validation/exp2.txt
#python main.py -c main_config_exp3.ini 2>&1 | tee ./logs/validation/exp3.txt
#python main.py -c main_config_exp4.ini 2>&1 | tee ./logs/validation/exp4.txt

#python main.py -c main_config_exp1.ini 2>&1 | tee ./logs/study/exp1.txt
#python main.py -c main_config_exp2.ini 2>&1 | tee ./logs/study/exp2.txt
#python main.py -c main_config_exp3.ini 2>&1 | tee ./logs/study/exp3.txt
#python main.py -c main_config_exp4.ini 2>&1 | tee ./logs/study/exp4.txt

python main.py -c main_config_exp1.ini 2>&1 | tee ./logs/testset_study/exp1.txt
python main.py -c main_config_exp2.ini 2>&1 | tee ./logs/testset_study/exp2.txt
python main.py -c main_config_exp3.ini 2>&1 | tee ./logs/testset_study/exp3.txt
python main.py -c main_config_exp4.ini 2>&1 | tee ./logs/testset_study/exp4.txt
