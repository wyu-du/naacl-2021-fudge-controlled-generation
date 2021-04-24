
export device_id=5

CUDA_VISIBLE_DEVICES=$device_id python -u evaluate_intent.py \
--ckpt ckpt/intent/intent_retrain_predictor/model.pth.tar \
--dataset_info ckpt/intent/intent_retrain_predictor/dataset_info \
--length_cutoff 60 \
--in_file dailydialog_data/dailydialog_gen_test.txt > intent_preds.log
#--model_path ckpt/formality/marian_finetune_fisher \


