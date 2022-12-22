#!/bin/sh

echo " "
echo "Enter number X to select option X:"
echo " "

MODELS_PATH="../save_model"
DATA_PATH="../Datasets"
echo "Pre-configured MODELS_PATH: ${MODELS_PATH}"
echo "Pre-configured DATA_PATH: ${DATA_PATH}"
echo " "

read -e -p "Dataset : (1) SYSU - (2) RegDB - (3) TWorld :" DATASET
Datasets=("SYSU" "RegDB" "TWorld")
DATASET=${Datasets[DATASET-1]}
echo $DATASET

read -e -p "Model : (1) concatenation - (2) unimodal - (3) transreid - (4) LightMBN :" MODEL
Model=("concatenation" "unimodal" "transreid" "LMBN")
MODEL=${Model[MODEL-1]}
echo $MODEL

if [ "$MODEL" != "concatenation" ]
then
    read -e -p "Trained using CIL ? (1) True - (2) False:" CIL
    Cil=("--CIL" " ")
    CIL=${Cil[CIL-1]}
    echo $CIL

    REID="VtoV"

else # For stacked generalization those parameters are not used, can be set to default
    read -e -p "Trained using ML-MDA ? (1) True - (2) False:" MLMDA
    Mlmda=("--ML_MDA" " ")
    MLMDA=${Mlmda[MLMDA-1]}
    echo $MLMDA

    REID="BtoB"
fi

read -e -p "Scenario evaluation - (1) normal (2) C (3) C* :" SCENARIO_EVAL
Scenario_eval=("normal" "C" "C*")
SCENARIO_EVAL=${Scenario_eval[SCENARIO_EVAL-1]}
echo $SCENARIO_EVAL

read -e -p "GPU value ? (0) - (1) - (2) - ... :" GPU
export CUDA_VISIBLE_DEVICES=$GPU
echo $CUDA_VISIBLE_DEVICES

echo


python test.py   --model=$MODEL \
                  --dataset=$DATASET \
                  --models_path=$MODELS_PATH \
                  --data_path=$DATA_PATH \
                  --scenario_eval=$SCENARIO_EVAL \
                  --reid=$REID \
                  $MLMDA \
                  $CIL;

