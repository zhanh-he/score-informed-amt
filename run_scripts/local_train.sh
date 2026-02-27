# !/bin/bash
python pytorch/train_score_inf.py model.type="hpt"\
  score_informed.method="direct" \
  model.input2=null model.input3=null \
  loss.loss_type="kim_bce_l1"

# --------- SCRR ---------
python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="scrr" \
  model.input2=onset model.input3=null \
  loss.loss_type="kim_bce_l1"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="scrr" \
  model.input2=onset model.input3=exframe \
  loss.loss_type="kim_bce_l1"

# --------- Dual Gated ---------
python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="dual_gated" \
  model.input2=onset model.input3=null \
  loss.loss_type="kim_bce_l1"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="dual_gated" \
  model.input2=onset model.input3=exframe \
  loss.loss_type="kim_bce_l1"

# --------- Note Editor ---------
python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="note_editor" \
  model.input2=onset model.input3=null \
  loss.loss_type="kim_bce_l1"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="note_editor" \
  model.input2=onset model.input3=exframe \
  loss.loss_type="kim_bce_l1"

# --------- BiLSTM ---------
python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="bilstm" \
  model.input2=onset model.input3=null \
  loss.loss_type="kim_bce_l1"

python pytorch/train_score_inf.py model.type="hpt" \
  score_informed.method="bilstm" \
  model.input2=onset model.input3=exframe \
  loss.loss_type="kim_bce_l1"

 # --------- HPPNet / Dynest ---------
python pytorch/train_score_inf.py model.type="hppnet"\
  score_informed.method="direct" \
  model.input2=null model.input3=null \
  loss.loss_type="kim_bce_l1"

python pytorch/train_score_inf.py model.type="dynest"\
  score_informed.method="direct" \
  model.input2=null model.input3=null \
  loss.loss_type="kim_bce_l1"
