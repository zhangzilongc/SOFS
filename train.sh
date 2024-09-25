for split_ in 0 1 2
do
  python3.8 main.py --init_method "tcp://localhost:9999" --device "0, 1, 2, 3, 4, 5, 7" \
  --cfg method_config/VISION_V1/Train/SOFS.yaml --prior_layer_pointer 7 \
  --opts NUM_GPUS 7 DATASET.split $split_ \
  TRAIN_SETUPS.epochs 10 TRAIN_SETUPS.TEST_SETUPS.epoch_test 10 TRAIN_SETUPS.TEST_SETUPS.train_miou 10 \
  TRAIN_SETUPS.TEST_SETUPS.test_state True TRAIN_SETUPS.TEST_SETUPS.val_state False \
  TRAIN.save_model False \
  DATASET.mix_sample_sampling False DATASET.mix_sample_sampling_prob 0. \
  DATASET.normal_sample_sampling_prob 0.3 \
  TRAIN.SOFS.smooth_r 1e5 \
  TRAIN.SOFS.feature_adaptation False\
  DATASET.name 'VISION_V1_ND' RNG_SEED 54
done

#python3.8 main.py --init_method "tcp://localhost:9999" --device "0, 1, 2, 3, 4, 5, 7" \
#--cfg method_config/VISION_V1/Train/SOFS.yaml --prior_layer_pointer 7 \
#--opts NUM_GPUS 7 DATASET.split 0 \
#TRAIN_SETUPS.epochs 50 TRAIN_SETUPS.TEST_SETUPS.epoch_test 10 TRAIN_SETUPS.TEST_SETUPS.train_miou 10 \
#TRAIN_SETUPS.TEST_SETUPS.test_state True TRAIN_SETUPS.TEST_SETUPS.val_state True \
#TRAIN.save_model False \
#DATASET.mix_sample_sampling False DATASET.mix_sample_sampling_prob 0. \
#DATASET.normal_sample_sampling_prob 0.3 \
#TRAIN.SOFS.smooth_r 1e5 \
#TRAIN.SOFS.feature_adaptation True \
#DATASET.defect_generation True \
#DATASET.name 'VISION_V1_ND' RNG_SEED 54

