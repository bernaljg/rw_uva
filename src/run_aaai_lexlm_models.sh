source myconda
mamba activate tf_220_1
python  /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/src/run_umls_classifier.py \
        --workspace_dp=/data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/data/lexlm_preds \
        --umls_version_dp=/data/Bodenreider_UMLS_DL/umls_reproduce/2020AB-ACTIVE \
        --umls_dl_dp=/data/Bodenreider_UMLS_DL/umls_reproduce/2020AB-ACTIVE/META_DL \
        --dataset_version_dp=/data/Bodenreider_UMLS_DL/umls_reproduce/2020AA-ACTIVE/NEGPOS1_WITHOUT_RAN \
        --train_dataset_dp=/data/Bodenreider_UMLS_DL/umls_reproduce/2020AA-ACTIVE/NEGPOS1_WITHOUT_RAN/LEARNING_DS/ALL \
        --test_dataset_fp=/data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/data/lexlm_2020AB_dev_set_2.RRF \
        --word_embedding="BioWordVec" --WordEmbVariant="BioWordVec" \
        --embedding_fp=/data/Bodenreider_UMLS_DL/extra/biowordvec.txt \
        --exp_flavor=1 --embedding_dim=200 --ConVariant="All_Triples" \
        --Model="TransE_SGD" --do_prep=false  --run_id="www_biowordvec_run1_lstm_cosine" \
        --lstm_attention="lstm" --n_epoch=100  --batch_size=8192  --use_shared_dropout_layer=false \
        --do_train=false --do_predict=true --do_predict_all=false --predict_test_dir_after_every_epoch=false \
        --load_IDs=true --generator_workers=8 --start_epoch_predict=100 --end_epoch_predict=100 \
        --logs_fp=/data/Bodenreider_UMLS_DL/Interns/Bernal/predict_2020AA-ACTIVE_MODEL_LEARNING_DS_ALL_8192b_200e_exp1_nodropout_www_biowordvec_run1_lstm_100_on.log