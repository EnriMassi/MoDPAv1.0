#!/usr/bin/env python
# coding: utf-8
import os, gc, names_generator, sys, re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from vae import VAE_bilayer

LABEL, DATASET = sys.argv[1:]

Input = pd.read_pickle(DATASET)
Input.fillna(0, inplace=True)
print(Input.values.max(), "|", Input.values.min(), "|", Input.shape)
# os.mkdir(LABEL)

Train, Test = train_test_split(Input, test_size=0.15)
# Train.to_pickle(os.path.join(LABEL,'train-data.pkl.gz'), compression='gzip')
# Train.to_pickle(os.path.join(LABEL,'train-data.pkl.gz'), compression='gzip')

# 2. Instantiate the VAE
vae = VAE_bilayer(
    original_dim = Train.shape[1],
    hidden_dim1  = 512,
    hidden_dim2  = 256,
    latent_dim   = 32,
    loss_type    = "cos+KL",   # alternatives: "mean_squared_error", "cosine_similarity", "RMSE+KL"
    rec_weight   = 1,
    dropout_rate = 0.25,
    multiply_by_og_dim = 1,
)

# 3. Compile with Adam
vae.compile(optimizer = optimizers.Adam(learning_rate=1e-3))

vae.fit(
    Train,
    epochs          = 999,
    batch_size      = 128,
    validation_data = (Test, None),  # targets are implicit
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True)]
)

results = vae.evaluate(
    Test,
    batch_size=256,
    # verbose=2
)
print(f"\nTest results:\n  Total loss = {results[0]:.4f}\n  Rec loss   = {results[1]:.4f}\n  KL loss    = {results[2]:.4f}\n")


# # — After training —
vae.build(input_shape=(None, Train.shape[1]))
savefld = os.path.join(LABEL,names_generator.generate_name())
vae.save(savefld)  # writes config.json, weights.h5, and saved_model/

# # — Later, or in a fresh session —
# # Rebuild + load weights
# del vae
# vae = VAE_bilayer.load_vae(
#     folder=savefld,
#     compile_kwargs={
#         "optimizer": optimizers.Adam(learning_rate=1e-3)
#     }
# )

# # Verify
# vae.evaluate(
#     Test,
#     batch_size=256,
#     verbose=2
# )

# # — Get encodings —
print('Get encodings...')
Test = Input.copy(deep=True)
del Input
ptms_list = [''.join(re.split(r'[][]',x)[:2]) for x in Test.index]
exps_list = list(Test.columns)

encoded       = np.array(vae.encode(Test.values))
reconstructed = np.array(vae.decode(encoded[0]))

fig,ax = plt.subplots(1,2,sharey=True,figsize=(16,9))
sns.heatmap(Test, vmin=0, vmax=1, center=0, 
            cmap='vlag', cbar=False, ax=ax[0])
sns.heatmap(reconstructed, vmin=0, vmax=1, center=0, 
            cmap='vlag', cbar=False, ax=ax[1])
fig.savefig(os.path.join(savefld,'reconstruction.png'), dpi=300, bbox_inches='tight')

encoded = pd.DataFrame(encoded[0,:,:], index=ptms_list)
encoded.to_pickle(os.path.join(savefld,'Latent-space.pkl.gz'), compression='gzip')

print('OG matrix size =', Test.shape)

# — garbage collector —
tf.keras.backend.clear_session()
_ = gc.collect()