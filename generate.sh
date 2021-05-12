
export device_id=5

CUDA_VISIBLE_DEVICES=$device_id python -u evaluate_intent.py \
--model_string microsoft/DialoGPT-medium \
--ckpt ckpt/intent/intent_retrain_predictor/model_best.pth.tar \
--dataset_info ckpt/intent/intent_retrain_predictor/dataset_info \
--length_cutoff 60 \
--in_file dailydialog_data/dailydialog_gen_test.txt > intent_dec_time.log
#--model_path ckpt/formality/marian_finetune_fisher \


