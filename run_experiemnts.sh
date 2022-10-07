# Is $DATASETS provided by the environment?
if [ -z "$DATASETS" ]; then
    echo "Plese set the $DATASETS environment variable to a path where datasets are/can be stored"
    exit 1
fi


# CORE50 does not have a pretrained model, so we cannot use internal replay.
# SE-CORE50 is used for comparison with internal replay.
almost_BIR="--feedback --prior=GMM --per-class --dg-gates --distill"
# CORE50 is 128x128, neccessitating additional reducing layers to reduce
# the size of the model.
core50_args="--reducing-layers 5 --depth 5"

# Run experiments 0 through 10
for EXP_ID in {0..10}
do
    echo "Experiment $EXP_ID"
    base="python3 main_cl.py --seed $EXP_ID --scenario=class --replay=generative --data-dir=$DATASETS"

    # =========================================================================
    # Generative Replay
    # =========================================================================
    GR=$base
    echo "GR_S-FMNIST"
    $GR --experiment=splitFMNIST  --tasks=5  > logs/GR_S-FMNIST_$EXP_ID.log
    echo "GR_S-CIFAR10"
    $GR --experiment=splitCIFAR10 --tasks=5  > logs/GR_S-CIFAR10_$EXP_ID.log
    echo "GR_S-CIFAR100"
    $GR --experiment=CIFAR100     --tasks=10 > logs/GR_S-CIFAR100_$EXP_ID.log
    echo "GR_S-CORE50"
    $GR --experiment=splitCORE50  --tasks=10 $core50_args  > logs/GR_S-CORE50_$EXP_ID.log


    # =========================================================================
    # Brain Inspired Replay
    # =========================================================================
    BIR="$base --brain-inspired"
    echo "BIR_S-FMNIST"
    $BIR --experiment=splitFMNIST  --tasks=5  > logs/BIR_S-FMNIST_$EXP_ID.log

    echo "BIR_S-CIFAR10"
    $BIR --experiment=splitCIFAR10 --tasks=5  > logs/BIR_S-CIFAR10_$EXP_ID.log

    echo "BIR_S-CIFAR100"
    $BIR  --experiment=CIFAR100     --tasks=10 > logs/BIR_S-CIFAR100_$EXP_ID.log

    echo "BIR_S-CORE50"
    $base $almost_BIR $core50_args --experiment=splitCORE50 --tasks=10 > logs/BIR_S-CORE50_$EXP_ID.log


    # =========================================================================
    # Brain Inspired Replay using ResNet-50 embedding
    # =========================================================================
    E_BIR="$base --extract-features True --brain-inspired --depth=0"
    echo "BIR_SE-CIFAR100"
    $E_BIR --experiment=CIFAR100    --tasks=10 > logs/BIR_SE-CIFAR100_$EXP_ID.log &

    echo "BIR_SE-CORE50"
    $E_BIR --experiment=splitCORE50 --tasks=10 > logs/BIR_SE-CORE50_$EXP_ID.log

    # =========================================================================
    # Brain Inspired Replay using ResNet-50 embedding
    # =========================================================================
    E_GR="$base --extract-features True"
    echo "GR_SE-CORE50"
    $E_GR --experiment=splitCORE50 --tasks=10  > logs/GR_SE-CORE50_$EXP_ID.log 

    echo "GR_SE-CIFAR100"
    $E_GR --experiment=CIFAR100     --tasks=10 > logs/GR_SE-CIFAR100_$EXP_ID.log &
done


