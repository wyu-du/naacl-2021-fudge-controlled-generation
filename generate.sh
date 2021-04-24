
export device_id=1

CUDA_VISIBLE_DEVICES=$device_id python -u evaluate_formality.py \
--ckpt ckpt/formality/predictor_gyafc_entertainment_music/model.pth.tar \
--dataset_info ckpt/formality/predictor_gyafc_entertainment_music/dataset_info \
--in_file formality_data/fisher_test_oracle.es \
--model_path ckpt/formality/marian_finetune_fisher > formality_preds.log


